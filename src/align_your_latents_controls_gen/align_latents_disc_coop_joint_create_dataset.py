import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

top_dir = Path(__file__).resolve().parent.parent.parent.parent.absolute()

sys.path.append(str(top_dir / "InterSem"))

# define_locations_for_hugging_face()

from src.utils.project_utils import plot_images_and_prompt, compute_FID_metrics, initialize_seed
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from src.align_your_latents_controls_gen.align_latents_pipeline_2_controls import StableDiffusionTwoControlsPipeline
from src.align_your_latents_controls_gen.unified_unet import UnetConditionModelUnified
from src.align_your_latents_controls_gen.align_latents_train_2_controls import collate_fn, negative_prompt, \
    CONTROLNET_PATH
from src.align_your_latents_controls_gen.align_latents_train_control_gen import dict_control_type_to_pref
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL, \
    UniPCMultistepScheduler
import torch
from src.dataset import InterSemDataset
from glob import glob
import numpy as np
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging
from typing import List, Tuple, Union
import torch.distributed as dist
import torch.multiprocessing as mp
from src.utils.project_utils import DistributedSamplerNoDuplicate
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate.utils import set_seed
from transformers import AutoTokenizer, CLIPTextModel
from PIL import Image
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
import re
from src.align_your_latents_controls_gen.align_latents_pipeline_control_gen import StableDiffusionControlGenPipeline

# from huggingface_hub import login
#
# login()

STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2-1-base"
DTYPE = torch.float16
RESOLUTION = 512
TOKENIZER_MAX_LENGTH = 77
SPLIT = "val"

logger = logging.get_logger(__name__)


def get_latest_checkpoint(checkpoint_prefix):
    list_dirs_checkpoint = [f for f in os.listdir(os.path.dirname(os.path.realpath(__file__))) if
                            re.search(f"{checkpoint_prefix}", f)]
    list_dirs_checkpoint = [Path(dir) for dir in list_dirs_checkpoint if Path(dir).is_dir()]
    list_dirs_checkpoint_sorted_by_creation_date = sorted(list_dirs_checkpoint, key=lambda file_name:
    int(file_name.name.replace("-best", "").split("-")[-1]), reverse=True)[:]
    checkpoint_path = list_dirs_checkpoint_sorted_by_creation_date[0]
    return checkpoint_path


def load_unet(control_type):
    checkpoint_path = get_latest_checkpoint(f"control-generation-checkpoint-{control_type}-[0-9]*$")
    print(f"loaded unet from: {checkpoint_path}")
    unet = UNet2DConditionModel.from_pretrained(STABLE_DIFFUSION_PATH, subfolder="unet")
    checkpoint = torch.load(checkpoint_path / "unet.pt")
    unet.load_state_dict(checkpoint, strict=True)
    unet.eval()
    return unet.to(DTYPE)


def load_vae(control_type):
    checkpoint_path = get_latest_checkpoint(f"vae-{control_type}-checkpoint-[0-9]*$")
    print(f"loaded vae from: {checkpoint_path}")
    vae = AutoencoderKL.from_pretrained(STABLE_DIFFUSION_PATH, subfolder="vae")
    checkpoint = torch.load(checkpoint_path / "vae.pt")
    vae.load_state_dict(checkpoint, strict=True)
    vae.eval()
    return vae.to(DTYPE)


def sample(rank,
           world_size,
           our_pipeline_control_1,
           our_pipeline_control_2,
           text_encoder,
           tokenizer,
           dataset,
           dist_url):
    torch.cuda.empty_cache()
    if world_size > 1:
        torch.cuda.set_device(rank)
    device = torch.device(rank)
    text_encoder.to(device)
    if world_size > 1:
        if rank == 0 and os.path.exists(dist_url):
            os.remove(dist_url)
        dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=dist_url)
        text_encoder = DDP(text_encoder, device_ids=[rank], find_unused_parameters=False)
    our_pipeline_control_1 = our_pipeline_control_1.to(rank)
    our_pipeline_control_2 = our_pipeline_control_2.to(rank)
    dataset.apply_hed.netNetwork = dataset.apply_hed.netNetwork.to(rank)
    if world_size > 1:
        distributed_sampler_train = DistributedSamplerNoDuplicate(dataset, shuffle=True)
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                sampler=distributed_sampler_train,
                                num_workers=0,
                                collate_fn=lambda examples: collate_fn(examples=examples,
                                                                       device=rank,
                                                                       tokenizer=tokenizer,
                                                                       text_encoder=text_encoder,
                                                                       tokenizer_max_length=77,
                                                                       zero_out_prob=0.0,
                                                                       control_type_1="depth",
                                                                       control_type_2="hed",
                                                                       ))
    else:
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=0,
                                # shuffle=True,
                                collate_fn=lambda examples: collate_fn(examples=examples,
                                                                       device=rank,
                                                                       tokenizer=tokenizer,
                                                                       text_encoder=text_encoder,
                                                                       tokenizer_max_length=77,
                                                                       zero_out_prob=0.0,
                                                                       control_type_1="depth",
                                                                       control_type_2="hed",
                                                                       ))
    dataset_dir = Path(f"{top_dir}/Depth_And_HED_dataset/{SPLIT}/")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"the number of samples is: {len(dataloader.dataset)}")
    num_images_per_pipeline = 10
    with torch.no_grad():
        prompts_input_control_1 = []
        prompts_input_control_2 = []
        for i, batch_val in enumerate(dataloader):
            for prompt in batch_val["text"]:
                prompt = prompt.replace("/", " ")
                if not (dataset_dir / (prompt + "_depth.png")).exists():
                    prompts_input_control_1.append(prompt)
                if not (dataset_dir / (prompt + "_hed.png")).exists():
                    prompts_input_control_2.append(prompt)
            if (i + 1) % num_images_per_pipeline == 0 and i > 0:
                with torch.autocast("cuda"):
                    if len(prompts_input_control_1) > 0:
                        generator = torch.Generator(device=rank).manual_seed(1)
                        pipeline_output = our_pipeline_control_1(
                            prompt=[dict_control_type_to_pref["depth"] + prompt.lower() for prompt in
                                    prompts_input_control_1],
                            negative_prompt=[negative_prompt] * len(prompts_input_control_1),
                            num_inference_steps=70,
                            generator=generator,
                            width=RESOLUTION,
                            height=RESOLUTION,
                        ).control_1
                        for j, control_1_generated_image in enumerate(pipeline_output):
                            control_1_generated_image.save(dataset_dir / (prompts_input_control_1[j] + "_depth.png"))
                    if len(prompts_input_control_2) > 0:
                        generator = torch.Generator(device=rank).manual_seed(1)
                        pipeline_output = our_pipeline_control_2(
                            prompt=[dict_control_type_to_pref["hed"] + prompt.lower() for prompt in
                                    prompts_input_control_2],
                            negative_prompt=[negative_prompt] * len(prompts_input_control_2),
                            num_inference_steps=70,
                            generator=generator,
                            width=RESOLUTION,
                            height=RESOLUTION,
                        ).control_1
                        for j, control_2_generated_image in enumerate(pipeline_output):
                            control_2_generated_image.save(dataset_dir / (prompts_input_control_2[j] + "_hed.png"))

                prompts_input_control_1.clear()
                prompts_input_control_2.clear()

    if world_size > 1:
        dist.destroy_process_group()


def prepare_pipelines():
    dataset = InterSemDataset(split=SPLIT, dataset_to_use="COCO", dataset_size=-1, sampling_mode=True)
    tokenizer = AutoTokenizer.from_pretrained(STABLE_DIFFUSION_PATH,
                                              subfolder="tokenizer",
                                              use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(STABLE_DIFFUSION_PATH, subfolder="text_encoder")

    vae_control_1 = load_vae(control_type="depth")
    vae_control_2 = load_vae(control_type="hed")
    unet_control_1 = load_unet(control_type="depth")
    unet_control_2 = load_unet(control_type="hed")

    our_pipeline_control_1 = StableDiffusionControlGenPipeline.from_pretrained(
        STABLE_DIFFUSION_PATH,
        unet=unet_control_1,
        vae=vae_control_1,
        safety_checker=None,
        torch_dtype=DTYPE
    )
    our_pipeline_control_1.scheduler = DDIMScheduler.from_config(our_pipeline_control_1.scheduler.config)

    our_pipeline_control_2 = StableDiffusionControlGenPipeline.from_pretrained(
        STABLE_DIFFUSION_PATH,
        unet=unet_control_2,
        vae=vae_control_2,
        safety_checker=None,
        torch_dtype=DTYPE
    )
    our_pipeline_control_2.scheduler = DDIMScheduler.from_config(our_pipeline_control_2.scheduler.config)

    return (our_pipeline_control_1,
            our_pipeline_control_2,
            text_encoder,
            tokenizer,
            dataset)


def main():
    dist_url = f"file://{top_dir}/DDP/DDP_FILE_{np.random.randint(10000)}"
    # initialize_seed(1)
    torch.multiprocessing.set_sharing_strategy('file_system')
    world_size = torch.cuda.device_count()
    (our_pipeline_control_1,
     our_pipeline_control_2,
     text_encoder,
     tokenizer,
     dataset) = prepare_pipelines()
    if world_size > 1:
        mp.spawn(sample,
                 args=(
                     world_size,
                     our_pipeline_control_1,
                     our_pipeline_control_2,
                     text_encoder,
                     tokenizer,
                     dataset,
                     dist_url),
                 nprocs=world_size,
                 join=True)
    else:
        sample(rank=0,
               world_size=1,
               our_pipeline_control_1=our_pipeline_control_1,
               our_pipeline_control_2=our_pipeline_control_2,
               text_encoder=text_encoder,
               tokenizer=tokenizer,
               dataset=dataset,
               dist_url=dist_url)


if __name__ == "__main__":
    main()
