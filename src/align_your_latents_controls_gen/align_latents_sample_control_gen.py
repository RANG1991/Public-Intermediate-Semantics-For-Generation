import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

top_dir = Path(__file__).resolve().parent.parent.parent.parent.absolute()

sys.path.append(str(top_dir / "InterSem"))

# define_locations_for_hugging_face()

from src.utils.project_utils import plot_images_and_prompt, compute_FID_metrics, get_last_checkpoint_dir, \
    initialize_seed
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from src.align_your_latents_controls_gen.align_latents_train_control_gen import collate_fn, \
    negative_prompt, dict_control_type_to_pref
from src.align_your_latents_controls_gen.align_latents_pipeline_control_gen import StableDiffusionControlGenPipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL, \
    UniPCMultistepScheduler
from diffusers import StableDiffusionControlNetPipeline as StableDiffusionControlNetPipelineDiffusers
import torch
from src.dataset import InterSemDataset
from glob import glob
import numpy as np
from diffusers.utils import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from src.utils.project_utils import DistributedSamplerNoDuplicate
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, CLIPTextModel
from PIL import Image
from enum import Enum

# from huggingface_hub import login
#
# login()

CONTROL_TYPE = "seg"
CONTROLNET_PATH = (f"thibaud/controlnet-sd21-{CONTROL_TYPE}-diffusers" if CONTROL_TYPE in ["hed", "depth"]
                   else f"thibaud/controlnet-sd21-ade20k-diffusers" if CONTROL_TYPE == "seg" else None)
STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2-1-base"
DTYPE = torch.float16
RESOLUTION = 512
TOKENIZER_MAX_LENGTH = 77
OVERWRITE = False
NUM_IMAGES_TO_CHECK = 20


class GEN_TYPE(Enum):
    SD = 1
    OUR_CONTROL = 2
    ORIG_CONTROL = 3


LIST_GEN_TYPES = [GEN_TYPE.ORIG_CONTROL]


def load_unet(control_type):
    checkpoint_path = get_last_checkpoint_dir(output_dir=Path(__file__).parent.absolute(),
                                              prefix=f"control-generation-checkpoint-{control_type}",
                                              get_best=True,
                                              accelerator=None)
    print(f"loaded unet from: {checkpoint_path}")
    unet = UNet2DConditionModel.from_pretrained(STABLE_DIFFUSION_PATH, subfolder="unet")
    checkpoint = torch.load(f"{checkpoint_path}/unet.pt")
    unet.load_state_dict(checkpoint, strict=True)
    unet.eval()
    return unet.to(DTYPE)


def load_vae(control_type):
    checkpoint_path = get_last_checkpoint_dir(output_dir=Path(__file__).parent.absolute(),
                                              prefix=f"vae-{control_type}-checkpoint",
                                              get_best=False,
                                              accelerator=None)
    print(f"loaded vae from: {checkpoint_path}")
    vae = AutoencoderKL.from_pretrained(STABLE_DIFFUSION_PATH, subfolder="vae")
    checkpoint = torch.load(f"{checkpoint_path}/vae.pt")
    vae.load_state_dict(checkpoint, strict=True)
    vae.eval()
    return vae.to(DTYPE)


def generate_from_prompts_our_text_to_image(our_pipeline_1_control,
                                            original_controlnet_pipeline,
                                            list_prompts_control_gen):
    list_prompts_control_gen_control_1 = [dict_control_type_to_pref[CONTROL_TYPE] + test_prompt.lower() for
                                          test_prompt in
                                          list_prompts_control_gen]
    with torch.autocast("cuda"):
        pipeline_output = our_pipeline_1_control(
            prompt=list_prompts_control_gen_control_1,
            negative_prompt=[negative_prompt] * len(list_prompts_control_gen),
            num_inference_steps=80,
            # guidance_scale=14,
            # generator=generator,
            width=RESOLUTION,
            height=RESOLUTION,
        )
        list_generated_control_gen_curr = [generated_control for generated_control in
                                           pipeline_output.control_1]

        # print(list_generated_control_gen_curr)
        # print(list_prompts_control_gen)

        generated_images_our_pipeline, _ = original_controlnet_pipeline(
            image=list_generated_control_gen_curr,
            prompt=list_prompts_control_gen,
            negative_prompt=[negative_prompt] * len(list_prompts_control_gen),
            num_inference_steps=80,
            # guidance_scale=14,
            # generator=generator,
            width=RESOLUTION,
            hieght=RESOLUTION,
            return_dict=False,
        )
    return list_generated_control_gen_curr, generated_images_our_pipeline


def save_generated_control_and_image(output_folder,
                                     list_generated_control_gen_curr,
                                     generated_images_our_pipeline,
                                     list_prompts,
                                     control_type):
    for (control_gen_pred, prompt, generated_image_our_pipeline) in (
            zip(list_generated_control_gen_curr, list_prompts, generated_images_our_pipeline)):
        pred_control = np.asarray(control_gen_pred)
        # if CONTROL_TYPE == "hed":
        #     gt_control = np.asarray(dataset.get_hed(generated_image_our_pipeline))
        # elif CONTROL_TYPE == "depth":
        #     gt_control = np.asarray(
        #         dataset.convert_image_to_depth_map(generated_image_our_pipeline, "")[0])
        # elif CONTROL_TYPE == "seg":
        #     gt_control = np.asarray(dataset.get_seg(generated_image_our_pipeline))
        imgs_comb = np.hstack([pred_control, np.asarray(generated_image_our_pipeline)])
        Image.fromarray(imgs_comb).save(output_folder /
                                        f"{control_type}_text_control_gen_{prompt.replace('/', '_')}.png")


def create_our_custom_images_and_control_from_prompts(output_folder, list_prompts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for control_type in ["seg", "depth", "hed"]:
        (original_control_net_pipeline,
         stable_diffusion_pipeline,
         our_pipeline_1_control,
         text_encoder,
         tokenizer,
         dataset,
         ) = prepare_pipelines(STABLE_DIFFUSION_PATH, control_type, CONTROLNET_PATH, DTYPE)
        list_generated_control_gen_curr, generated_images_our_pipeline = (
            generate_from_prompts_our_text_to_image(
                our_pipeline_1_control.to(device),
                original_control_net_pipeline.to(device),
                list_prompts))
        save_generated_control_and_image(output_folder,
                                         list_generated_control_gen_curr,
                                         generated_images_our_pipeline,
                                         list_prompts,
                                         control_type)


def sample(rank, world_size,
           our_pipeline_1_control,
           # our_controlnet_pipeline,
           original_controlnet_pipeline,
           stable_diffusion_pipeline,
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
    our_pipeline_1_control = our_pipeline_1_control.to(rank)
    # our_controlnet_pipeline = our_controlnet_pipeline.to(rank)
    original_controlnet_pipeline = original_controlnet_pipeline.to(rank)
    stable_diffusion_pipeline = stable_diffusion_pipeline.to(rank)
    dataset.apply_hed.netNetwork = dataset.apply_hed.netNetwork.to(rank)
    # dataset = torch.utils.data.Subset(dataset, list(range(0, 300, 15)))
    if world_size > 1:
        distributed_sampler_train = DistributedSamplerNoDuplicate(dataset, shuffle=False)
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
                                                                       control_type=CONTROL_TYPE,
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
                                                                       control_type=CONTROL_TYPE,
                                                                       ))
    if GEN_TYPE.OUR_CONTROL in LIST_GEN_TYPES:
        text_control_gen = Path(f"{top_dir}/InterSem/test_imgs/{CONTROL_TYPE}_text_control_gen/")
        text_control_gen.mkdir(parents=True, exist_ok=True)
        text_control_gen_image_and_control = Path(f"{top_dir}/InterSem/test_imgs/{CONTROL_TYPE}_image_and_control/")
        text_control_gen_image_and_control.mkdir(parents=True, exist_ok=True)
    if GEN_TYPE.ORIG_CONTROL in LIST_GEN_TYPES:
        original_depth_dir = Path(f"{top_dir}/InterSem/test_imgs/{CONTROL_TYPE}_original_control_gen/")
        original_depth_dir.mkdir(parents=True, exist_ok=True)
    if GEN_TYPE.SD in LIST_GEN_TYPES:
        original_SD_dir = Path(f"{top_dir}/InterSem/test_imgs/original_SD/")
        original_SD_dir.mkdir(parents=True, exist_ok=True)

    original_image_dir = Path(f"{top_dir}/InterSem/test_imgs/original/")
    original_image_dir.mkdir(parents=True, exist_ok=True)

    print(f"the number of samples is: {len(dataloader.dataset)}")
    num_images_per_pipeline = (1 if GEN_TYPE.ORIG_CONTROL in LIST_GEN_TYPES else 10)
    with (torch.no_grad()):
        titles_to_plot = []
        list_ground_truth_control_gen = []
        list_ground_truth_images_not_exist = []
        list_prompts_control_gen = []
        list_prompts_original_SD = []
        list_prompts_original_controlnet = []
        files_text_control_gen_list = []
        files_text_control_gen_list_image_and_control = []
        files_original_depth_list = []
        files_original_SD_list = []
        files_original_image_list = []
        images_to_plot = []
        for i, batch_val in enumerate(dataloader):
            print("memory: ", torch.cuda.mem_get_info()[0] / (10 ** 9), flush=True)
            torch.cuda.empty_cache()
            print(f"sample number : {i}", flush=True)
            test_prompts_to_check = batch_val["text"]
            for test_prompt_to_check in test_prompts_to_check:
                test_prompt_to_check = test_prompt_to_check.replace("/", "_")

                if GEN_TYPE.OUR_CONTROL in LIST_GEN_TYPES:
                    file_text_control_gen = Path(text_control_gen / f"{test_prompt_to_check}.png")
                    file_text_control_gen_image_and_control = Path(text_control_gen_image_and_control
                                                                   / f"{test_prompt_to_check}.png")
                    if not file_text_control_gen.exists() or OVERWRITE:
                        list_prompts_control_gen.extend(batch_val["text"])
                        files_text_control_gen_list.append(file_text_control_gen)
                        files_text_control_gen_list_image_and_control.append(file_text_control_gen_image_and_control)
                    else:
                        if (i + 1) < NUM_IMAGES_TO_CHECK:
                            titles_to_plot.append(file_text_control_gen.stem)
                            images_to_plot.append(Image.open(file_text_control_gen.absolute()))

                if GEN_TYPE.ORIG_CONTROL in LIST_GEN_TYPES:
                    file_original_depth_dir = Path(original_depth_dir / f"{test_prompt_to_check}.png")
                    if not file_original_depth_dir.exists():
                        files_original_depth_list.append(file_original_depth_dir)
                        list_ground_truth_control_gen.extend(batch_val["control"])
                        list_prompts_original_controlnet.extend(batch_val["text"])
                    else:
                        if (i + 1) < NUM_IMAGES_TO_CHECK:
                            titles_to_plot.append(file_original_depth_dir.stem)
                            images_to_plot.append(Image.open(file_original_depth_dir.absolute()))

                if GEN_TYPE.SD in LIST_GEN_TYPES:
                    file_original_SD_dir = Path(original_SD_dir / f"{test_prompt_to_check}.png")
                    if not file_original_SD_dir.exists():
                        list_prompts_original_SD.extend(batch_val["text"])
                        files_original_SD_list.append(file_original_SD_dir)
                    else:
                        if (i + 1) < NUM_IMAGES_TO_CHECK:
                            titles_to_plot.append(file_original_SD_dir.stem)
                            images_to_plot.append(Image.open(file_original_SD_dir.absolute()))

                file_original_image_dir = Path(original_image_dir / f"{test_prompt_to_check}.png")
                if not file_original_image_dir.exists():
                    list_ground_truth_images_not_exist.extend(batch_val["pixel_values"])
                    files_original_image_list.append(file_original_image_dir)
                else:
                    if (i + 1) < NUM_IMAGES_TO_CHECK:
                        titles_to_plot.append(file_original_image_dir.stem)
                        images_to_plot.append(Image.open(file_original_image_dir.absolute()))

            if (i + 1) % num_images_per_pipeline == 0 and i > 0:

                with torch.autocast("cuda"):
                    if len(files_text_control_gen_list) > 0 and GEN_TYPE.OUR_CONTROL in LIST_GEN_TYPES:
                        # generator = torch.Generator(device=rank).manual_seed(1)
                        (list_generated_control_gen_curr,
                         generated_images_our_pipeline) = generate_from_prompts_our_text_to_image(
                            our_pipeline_1_control,
                            original_controlnet_pipeline,
                            list_prompts_control_gen
                        )
                        for (control_gen_pred,
                             generated_image_our_pipeline,
                             file_text_control_gen,
                             file_text_control_gen_image_and_control) in zip(
                            list_generated_control_gen_curr,
                            generated_images_our_pipeline,
                            files_text_control_gen_list,
                            files_text_control_gen_list_image_and_control):
                            img_comb = np.hstack([control_gen_pred, np.asarray(generated_image_our_pipeline)])
                            Image.fromarray(img_comb).save(file_text_control_gen_image_and_control.absolute())
                            generated_image_our_pipeline.resize((256, 256)).save(file_text_control_gen.absolute())
                            titles_to_plot.extend([file_text_control_gen.stem] * 2)
                            images_to_plot.append(control_gen_pred)
                            images_to_plot.append(generated_image_our_pipeline)

                    if len(files_original_depth_list) > 0 and GEN_TYPE.ORIG_CONTROL in LIST_GEN_TYPES:
                        # generator = torch.Generator(device=rank).manual_seed(1)
                        # print(list_ground_truth_control_gen)
                        generated_images_original_controlnet, _ = original_controlnet_pipeline(
                            image=list_ground_truth_control_gen,
                            prompt=list_prompts_original_controlnet,
                            negative_prompt=[negative_prompt] * len(list_prompts_original_controlnet),
                            num_inference_steps=80,
                            # guidance_scale=14,
                            # generator=generator,
                            width=RESOLUTION,
                            hieght=RESOLUTION,
                            return_dict=False,
                        )
                        for (control_gen_ground_truth,
                             generated_image_original_controlnet,
                             file_original_depth) in zip(
                            list_ground_truth_control_gen,
                            generated_images_original_controlnet,
                            files_original_depth_list):
                            generated_image_original_controlnet.resize((256, 256)).save(file_original_depth.absolute())
                            titles_to_plot.extend([file_original_depth.stem] * 2)
                            images_to_plot.append(control_gen_ground_truth)
                            images_to_plot.append(generated_image_original_controlnet)

                    if len(files_original_SD_list) > 0 and GEN_TYPE.SD in LIST_GEN_TYPES:
                        print(f"the size of prompts: {len(list_prompts_original_SD)}", flush=True)
                        # generator = torch.Generator(device=rank).manual_seed(1)
                        output_images_original_SD, _ = stable_diffusion_pipeline(
                            prompt=list_prompts_original_SD,
                            negative_prompt=[negative_prompt] * len(list_prompts_original_SD),
                            num_inference_steps=80,
                            # guidance_scale=14,
                            # generator=generator,
                            width=RESOLUTION,
                            height=RESOLUTION,
                            return_dict=False,
                        )
                        for output_image_original_SD, file_original_SD_dir in zip(output_images_original_SD,
                                                                                  files_original_SD_list):
                            output_image_original_SD.resize((256, 256)).save(file_original_SD_dir.absolute())
                            titles_to_plot.append(file_original_SD_dir.stem)
                            images_to_plot.append(output_image_original_SD)

                    if len(files_original_image_list) > 0:
                        for original_image, file_original_image_dir in zip(list_ground_truth_images_not_exist,
                                                                           files_original_image_list):
                            original_image.resize((256, 256)).save(file_original_image_dir.absolute())
                            titles_to_plot.append(file_original_image_dir.stem)
                            images_to_plot.append(original_image)

                    if (i + 1) == NUM_IMAGES_TO_CHECK and len(images_to_plot) > 0:
                        plot_images_and_prompt(images=images_to_plot,
                                               titles=titles_to_plot,
                                               prompt="test images example")
                        titles_to_plot.clear()
                        images_to_plot.clear()

                titles_to_plot.clear()
                list_ground_truth_control_gen.clear()
                list_ground_truth_images_not_exist.clear()
                list_prompts_control_gen.clear()
                list_prompts_original_SD.clear()
                list_prompts_original_controlnet.clear()
                files_text_control_gen_list.clear()
                files_text_control_gen_list_image_and_control.clear()
                files_original_depth_list.clear()
                files_original_SD_list.clear()
                files_original_image_list.clear()
                images_to_plot.clear()

    if world_size > 1:
        dist.destroy_process_group()


def prepare_pipelines(stable_diffusion_path, control_type, controlnet_path, dtype):
    stable_diffusion_pipeline = StableDiffusionPipeline.from_pretrained(stable_diffusion_path,
                                                                        safety_checker=None,
                                                                        torch_dtype=dtype,
                                                                        # force_download=True
                                                                        )
    stable_diffusion_pipeline.scheduler = DDIMScheduler.from_config(
        stable_diffusion_pipeline.scheduler.config)

    original_control_net_pipeline = (
        StableDiffusionControlNetPipelineDiffusers.from_pretrained(
            stable_diffusion_path,
            controlnet=ControlNetModel.from_pretrained(controlnet_path),
            safety_checker=None,
            torch_dtype=dtype,
        ))
    original_control_net_pipeline.scheduler = DDIMScheduler.from_config(
        original_control_net_pipeline.scheduler.config)

    dataset = InterSemDataset(split="val",
                              dataset_to_use="COCO",
                              dataset_size=-1,
                              only_gt_image_and_prompt=(False if GEN_TYPE.ORIG_CONTROL in LIST_GEN_TYPES else True))
    tokenizer = AutoTokenizer.from_pretrained(stable_diffusion_path,
                                              subfolder="tokenizer",
                                              use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_path, subfolder="text_encoder")

    vae_control_1 = load_vae(control_type=control_type)
    unet_control_1 = load_unet(control_type=control_type)

    our_pipeline_1_control = StableDiffusionControlGenPipeline.from_pretrained(
        stable_diffusion_path,
        unet=unet_control_1,
        vae=vae_control_1,
        safety_checker=None,
        torch_dtype=dtype
    )
    our_pipeline_1_control.scheduler = DDIMScheduler.from_config(our_pipeline_1_control.scheduler.config)

    return (original_control_net_pipeline,
            stable_diffusion_pipeline,
            our_pipeline_1_control,
            text_encoder,
            tokenizer,
            dataset)


def main():
    dist_url = f"file://{top_dir}/DDP/DDP_FILE_{np.random.randint(10000)}"
    # initialize_seed(1)
    torch.multiprocessing.set_sharing_strategy('file_system')
    world_size = torch.cuda.device_count()
    (original_control_net_pipeline,
     stable_diffusion_pipeline,
     our_pipeline_1_control,
     text_encoder,
     tokenizer,
     dataset,
     ) = prepare_pipelines(STABLE_DIFFUSION_PATH, CONTROL_TYPE, CONTROLNET_PATH, DTYPE)
    if world_size > 1:
        mp.spawn(sample,
                 args=(
                     world_size,
                     our_pipeline_1_control,
                     original_control_net_pipeline,
                     stable_diffusion_pipeline,
                     text_encoder,
                     tokenizer,
                     dataset,
                     dist_url),
                 nprocs=world_size,
                 join=True)
    else:
        sample(rank=0,
               world_size=1,
               original_controlnet_pipeline=original_control_net_pipeline,
               stable_diffusion_pipeline=stable_diffusion_pipeline,
               our_pipeline_1_control=our_pipeline_1_control,
               text_encoder=text_encoder,
               tokenizer=tokenizer,
               dataset=dataset,
               dist_url=dist_url)


if __name__ == "__main__":
    # create_our_custom_images_and_control_from_prompts(top_dir / "InterSem/test_imgs/comp_for_paper",
    #                                                   [
    #                                                       "A brown bear walking near a tree on a field.",
    #                                                       "A red plate topped with a chocolate pastry.",
    #                                                       "a large group of people in a giant room with giant umbrellas",
    #                                                       "A clock tower next to a building in a city.",
    #                                                       "Old rusty fire hydrant on a street with white paint on top.",
    #                                                       "Cars and trucks driving under the underpass with "
    #                                                       "street signs directing them where to go.",
    #                                                       "A lady sitting in a chair in a hotel room.",
    #                                                   ])
    main()
