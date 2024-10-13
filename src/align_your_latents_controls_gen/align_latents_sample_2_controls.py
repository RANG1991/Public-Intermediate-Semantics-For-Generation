import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

top_dir = Path(__file__).resolve().parent.parent.parent.parent.absolute()

sys.path.append(str(top_dir / "InterSem"))

# define_locations_for_hugging_face()

from src.utils.project_utils import plot_images_and_prompt, compute_FID_metrics, initialize_seed, \
    get_last_checkpoint_dir
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from src.align_your_latents_controls_gen.align_latents_pipeline_2_controls import StableDiffusionTwoControlsPipeline
from src.align_your_latents_controls_gen.unified_unet import UnetConditionModelUnified
from src.align_your_latents_controls_gen.cross_attn_unified_unet import UnetConditionCrossAttnUnified
from src.align_your_latents_controls_gen.align_latents_train_2_controls import collate_fn, negative_prompt
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
from src.utils.calculate_FID import get_common_files_in_folders
from os.path import isfile, join
from enum import Enum

# from huggingface_hub import login
#
# login()
CONTROL_TYPE_1 = "depth"
CONTROL_TYPE_2 = "hed"
CONTROLNET_PATH_1 = (f"thibaud/controlnet-sd21-{CONTROL_TYPE_1}-diffusers" if CONTROL_TYPE_1 in ["hed", "depth"]
                     else f"thibaud/controlnet-sd21-ade20k-diffusers" if CONTROL_TYPE_1 == "seg" else None)
CONTROLNET_PATH_2 = (f"thibaud/controlnet-sd21-{CONTROL_TYPE_2}-diffusers" if CONTROL_TYPE_2 in ["hed", "depth"]
                     else f"thibaud/controlnet-sd21-ade20k-diffusers" if CONTROL_TYPE_2 == "seg" else None)

STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2-1-base"
DTYPE = torch.float32
RESOLUTION = 512
TOKENIZER_MAX_LENGTH = 77
OVERWRITE = True


class GEN_TYPE(Enum):
    SD = 1
    OUR_CONTROL = 2
    ORIG_CONTROL = 3


LIST_GEN_TYPES = [GEN_TYPE.OUR_CONTROL]


def load_multi_controlnet(control_type_1, control_type_2):
    checkpoint_path = get_last_checkpoint_dir(output_dir=Path(__file__).parent.absolute(),
                                              prefix=f"multi-controlnet-generation-checkpoint-{control_type_1}-{control_type_2}",
                                              get_best=True,
                                              accelerator=None)
    # checkpoint_path = Path("/sci/labs/sagieb/ranga/InterSem/src/align_your_latents_controls_gen/"
    #                        "multi-controlnet-generation-checkpoint-depth-hed-70/")
    print(f"loaded multi controlnet from: {checkpoint_path}")
    controlnet = MultiControlNetModel([ControlNetModel.from_pretrained(controlnet_checkpoint) for
                                       controlnet_checkpoint in CONTROLNET_PATH])
    checkpoint = torch.load(f"{checkpoint_path}/controlnet.pt")
    controlnet.load_state_dict(checkpoint, strict=True)
    controlnet.eval()
    return controlnet.to(DTYPE)


def load_unet_unified(control_type_1, control_type_2, unet_control_1, unet_control_2):
    checkpoint_path = get_last_checkpoint_dir(output_dir=Path(__file__).parent.absolute(),
                                              prefix=f"control-generation-checkpoint-{control_type_1}-{control_type_2}",
                                              get_best=True,
                                              accelerator=None)
    # checkpoint_path = Path("/mnt/proj3/dd-24-37/ran/control-generation-checkpoint-depth-hed-13600/")
    print(f"loaded unified unet from: {checkpoint_path}")
    unet_unified = UnetConditionModelUnified([unet_control_1, unet_control_2])
    checkpoint = torch.load(f"{checkpoint_path}/unet.pt")
    unet_unified.load_state_dict(checkpoint, strict=True)
    unet_unified.eval()
    return unet_unified.to(DTYPE)


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


def sample(rank, world_size,
           our_pipeline_2_controls,
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
    our_pipeline_2_controls = our_pipeline_2_controls.to(rank)
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
                                                                       control_type_1=CONTROL_TYPE_1,
                                                                       control_type_2=CONTROL_TYPE_2,
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
                                                                       control_type_1=CONTROL_TYPE_1,
                                                                       control_type_2=CONTROL_TYPE_2,
                                                                       ))
    if GEN_TYPE.OUR_CONTROL in LIST_GEN_TYPES:
        text_2_controls = Path(f"{top_dir}/InterSem/test_imgs/{CONTROL_TYPE_1}_{CONTROL_TYPE_2}_text_control_gen/")
        text_2_controls.mkdir(parents=True, exist_ok=True)
        text_control_gen_image_and_controls = Path(
            f"{top_dir}/InterSem/test_imgs/{CONTROL_TYPE_1}_{CONTROL_TYPE_2}_image_and_control")
        text_control_gen_image_and_controls.mkdir(parents=True, exist_ok=True)
    if GEN_TYPE.ORIG_CONTROL in LIST_GEN_TYPES:
        original_depth_dir = Path(f"{top_dir}/InterSem/test_imgs/original_2_controls/")
        original_depth_dir.mkdir(parents=True, exist_ok=True)
    if GEN_TYPE.SD in LIST_GEN_TYPES:
        original_SD_dir = Path(f"{top_dir}/InterSem/test_imgs/original_SD/")
        original_SD_dir.mkdir(parents=True, exist_ok=True)

    original_image_dir = Path(f"{top_dir}/InterSem/test_imgs/original/")
    original_image_dir.mkdir(parents=True, exist_ok=True)

    print(f"the number of samples is: {len(dataloader.dataset)}")
    num_images_per_pipeline = (1 if GEN_TYPE.ORIG_CONTROL in LIST_GEN_TYPES else 2)
    with (torch.no_grad()):
        titles_to_plot = []
        list_ground_truth_controls_1 = []
        list_ground_truth_controls_2 = []
        list_ground_truth_images_not_exist = []
        list_prompts_1 = []
        list_prompts_2 = []
        list_prompts_unified = []
        list_prompts_original_SD = []
        list_prompts_original_controlnet = []
        list_prompt_embeds_unified = []
        list_prompt_embeds_neg_unified = []
        files_text_2_controls_list = []
        files_text_2_controls_list_image_and_controls = []
        files_original_depth_list = []
        files_original_SD_list = []
        files_original_image_list = []
        images_to_plot = []
        for i, batch_val in enumerate(dataloader):
            print(f"sample number : {i}")
            test_prompts_to_check = batch_val["text"]
            for test_prompt_to_check in test_prompts_to_check:
                test_prompt_to_check = test_prompt_to_check.replace("/", "_")

                if GEN_TYPE.OUR_CONTROL in LIST_GEN_TYPES:
                    file_text_2_controls = Path(text_2_controls / f"{test_prompt_to_check}.png")
                    file_text_control_gen_image_and_control = Path(text_control_gen_image_and_controls
                                                                   / f"{CONTROL_TYPE_1}_{CONTROL_TYPE_2}"
                                                                     f"_text_control_gen_{test_prompt_to_check}.png")
                    if not file_text_2_controls.exists() or OVERWRITE:
                        list_prompts_1.extend(batch_val["text"])
                        list_prompts_2.extend(batch_val["text"])
                        list_prompts_unified.extend(batch_val["text"])
                        list_prompt_embeds_unified.append(batch_val["txt_embeddings_unet_unified"])
                        list_prompt_embeds_neg_unified.append(batch_val["txt_embeddings_negative"])
                        files_text_2_controls_list.append(file_text_2_controls)
                        files_text_2_controls_list_image_and_controls.append(file_text_control_gen_image_and_control)
                    else:
                        titles_to_plot.append(file_text_2_controls.stem)
                        images_to_plot.append(Image.open(file_text_2_controls.absolute()))

                if GEN_TYPE.ORIG_CONTROL in LIST_GEN_TYPES:
                    file_original_depth_dir = Path(original_depth_dir / f"{test_prompt_to_check}.png")
                    if not file_original_depth_dir.exists():
                        files_original_depth_list.append(file_original_depth_dir)
                        list_ground_truth_controls_1.extend(batch_val["control_1"])
                        list_ground_truth_controls_2.extend(batch_val["control_2"])
                        list_prompts_original_controlnet.extend(batch_val["text"])
                    else:
                        titles_to_plot.append(file_original_depth_dir.stem)
                        images_to_plot.append(Image.open(file_original_depth_dir.absolute()))

                if GEN_TYPE.SD in LIST_GEN_TYPES:
                    file_original_SD_dir = Path(original_SD_dir / f"{test_prompt_to_check}.png")
                    if not file_original_SD_dir.exists():
                        list_prompts_original_SD.extend(batch_val["text"])
                        files_original_SD_list.append(file_original_SD_dir)
                    else:
                        titles_to_plot.append(file_original_SD_dir.stem)
                        images_to_plot.append(Image.open(file_original_SD_dir.absolute()))

                file_original_image_dir = Path(original_image_dir / f"{test_prompt_to_check}.png")
                if not file_original_image_dir.exists():
                    list_ground_truth_images_not_exist.extend(batch_val["pixel_values"])
                    files_original_image_list.append(file_original_image_dir)
                else:
                    titles_to_plot.append(file_original_image_dir.stem)
                    images_to_plot.append(Image.open(file_original_image_dir.absolute()))

            if (i + 1) % num_images_per_pipeline == 0:
                with torch.autocast("cuda"):
                    if len(files_text_2_controls_list) > 0 and GEN_TYPE.OUR_CONTROL in LIST_GEN_TYPES:
                        list_prompts_1 = [dict_control_type_to_pref[CONTROL_TYPE_1] + test_prompt.lower() for
                                          test_prompt in
                                          list_prompts_1]
                        list_prompts_2 = [dict_control_type_to_pref[CONTROL_TYPE_2] + test_prompt.lower() for
                                          test_prompt in
                                          list_prompts_2]
                        prompts_input = [list(prompts_input) for prompts_input in
                                         zip(list_prompts_1, list_prompts_2, list_prompts_unified)][0]
                        # print(prompts_input)

                        generator = torch.Generator(device=rank).manual_seed(1)
                        pipeline_output = our_pipeline_2_controls(
                            prompts=prompts_input,
                            negative_prompt=negative_prompt,
                            num_inference_steps=80,
                            # guidance_scale=14,
                            generator=generator,
                            width=RESOLUTION,
                            height=RESOLUTION,
                        )
                        list_generated_controls_1_curr = [generated_control for generated_control in
                                                          pipeline_output.control_1]
                        list_generated_controls_2_curr = [generated_control for generated_control in
                                                          pipeline_output.control_2]
                        images_input_generated_controlnet = [list(images_input_tuple) for images_input_tuple in
                                                             zip(list_generated_controls_1_curr,
                                                                 list_generated_controls_2_curr)]

                        prompts_embeds_input_original_controlnet = torch.cat(list_prompt_embeds_unified).to(rank)
                        prompts_embeds_neg_input_original_controlnet = torch.cat(list_prompt_embeds_neg_unified).to(
                            rank)
                        # generator = torch.Generator(device=rank).manual_seed(1)
                        generated_images_our_pipeline, _ = original_controlnet_pipeline(
                            image=images_input_generated_controlnet,
                            prompt=list_prompts_unified,
                            negative_prompt=[negative_prompt] * len(list_prompts_unified),
                            # controlnet_conditioning_scale=[1.0, 1.0],
                            num_inference_steps=80,
                            # guidance_scale=14,
                            generator=generator,
                            width=RESOLUTION,
                            height=RESOLUTION,
                            return_dict=False,
                        )
                        for (control_1_pred,
                             control_2_pred,
                             generated_image_our_pipeline,
                             file_text_2_controls,
                             file_text_control_gen_image_and_control) in zip(
                            list_generated_controls_1_curr,
                            list_generated_controls_2_curr,
                            generated_images_our_pipeline,
                            files_text_2_controls_list,
                            files_text_2_controls_list_image_and_controls):
                            # gt_control_1 = np.asarray(
                            #     dataset.convert_image_to_depth_map(generated_image_our_pipeline, "")[0])
                            # gt_control_2 = np.asarray(dataset.get_hed(generated_image_our_pipeline))
                            pred_control_1 = np.asarray(control_1_pred)
                            pred_control_2 = np.asarray(control_2_pred)
                            imgs_comb = np.hstack([
                                # gt_control_1,
                                pred_control_1,
                                # gt_control_2,
                                pred_control_2,
                                np.asarray(generated_image_our_pipeline)
                            ])
                            try:
                                Image.fromarray(imgs_comb).save(file_text_control_gen_image_and_control.absolute())
                            except OSError as exc:
                                if exc.errno == 36:
                                    print(exc)
                            generated_image_our_pipeline.resize((256, 256)).save(file_text_2_controls.absolute())
                            titles_to_plot.extend([file_text_2_controls.stem] * 3)
                            images_to_plot.append(control_1_pred)
                            images_to_plot.append(control_2_pred)
                            images_to_plot.append(generated_image_our_pipeline)

                    if len(files_original_depth_list) > 0 and GEN_TYPE.ORIG_CONTROL in LIST_GEN_TYPES:
                        images_input_ground_truth_controlnet = [list(images_input_tuple) for images_input_tuple in
                                                                zip(list_ground_truth_controls_1,
                                                                    list_ground_truth_controls_2)]

                        generator = torch.Generator(device=rank).manual_seed(1)
                        generated_images_original_controlnet, _ = original_controlnet_pipeline(
                            image=images_input_ground_truth_controlnet,
                            prompt=list_prompts_original_controlnet,
                            negative_prompt=[negative_prompt] * len(list_prompts_original_controlnet),
                            # controlnet_conditioning_scale=[0.5, 0.5],
                            num_inference_steps=80,
                            # guidance_scale=14,
                            generator=generator,
                            width=RESOLUTION,
                            height=RESOLUTION,
                            return_dict=False,
                        )
                        for (control_1_ground_truth,
                             control_2_ground_truth,
                             generated_image_original_controlnet,
                             file_original_depth) in zip(
                            list_ground_truth_controls_1,
                            list_ground_truth_controls_2,
                            generated_images_original_controlnet,
                            files_original_depth_list):
                            generated_image_original_controlnet.resize((256, 256)).save(file_original_depth.absolute())
                            titles_to_plot.extend([file_original_depth.stem] * 3)
                            images_to_plot.append(control_1_ground_truth)
                            images_to_plot.append(control_2_ground_truth)
                            images_to_plot.append(generated_image_original_controlnet)

                    if len(files_original_SD_list) > 0 and GEN_TYPE.SD in LIST_GEN_TYPES:
                        generator = torch.Generator(device=rank).manual_seed(1)
                        output_images_original_SD, _ = stable_diffusion_pipeline(
                            prompt_embeds=list_prompts_original_SD,
                            negative_prompt_embeds=[negative_prompt] * len(list_prompts_original_SD),
                            num_inference_steps=80,
                            # guidance_scale=14,
                            generator=generator,
                            width=RESOLUTION,
                            height=RESOLUTION,
                            return_dict=False,
                        )

                        # generator = torch.Generator(device=rank).manual_seed(1)
                        # output_images_original_SD, _ = original_controlnet_pipeline(
                        #     image=[dataset.convert_image_to_depth_map(output_images_original_SD[0], "")[0],
                        #            dataset.get_hed(output_images_original_SD[0])],
                        #     prompt=list_prompts_original_controlnet,
                        #     negative_prompt=[negative_prompt] * len(list_prompts_original_controlnet),
                        #     controlnet_conditioning_scale=[0.5, 0.5],
                        #     num_inference_steps=80,
                        #     # guidance_scale=14,
                        #     generator=generator,
                        #     width=RESOLUTION,
                        #     height=RESOLUTION,
                        #     return_dict=False,
                        # )

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

                    if (i + 1) == 20 and len(images_to_plot) > 0:
                        plot_images_and_prompt(images=images_to_plot,
                                               titles=titles_to_plot,
                                               prompt="test images example")
                        images_to_plot.clear()
                        titles_to_plot.clear()

                titles_to_plot.clear()
                list_ground_truth_controls_1.clear()
                list_ground_truth_controls_2.clear()
                list_ground_truth_images_not_exist.clear()
                list_prompts_1.clear()
                list_prompts_2.clear()
                list_prompts_unified.clear()
                list_prompts_original_SD.clear()
                list_prompts_original_controlnet.clear()
                list_prompt_embeds_unified.clear()
                list_prompt_embeds_neg_unified.clear()
                files_text_2_controls_list.clear()
                files_text_2_controls_list_image_and_controls.clear()
                files_original_depth_list.clear()
                files_original_SD_list.clear()
                files_original_image_list.clear()
                images_to_plot.clear()

    if world_size > 1:
        dist.destroy_process_group()


def prepare_pipelines():
    stable_diffusion_pipeline = StableDiffusionPipeline.from_pretrained(STABLE_DIFFUSION_PATH,
                                                                        safety_checker=None,
                                                                        torch_dtype=DTYPE,
                                                                        # force_download=True
                                                                        )
    stable_diffusion_pipeline.scheduler = DDIMScheduler.from_config(stable_diffusion_pipeline.scheduler.config)

    controlnet_checkpoint_paths = [CONTROLNET_PATH_1, CONTROLNET_PATH_2]
    controlnet_list = []
    for checkpoint_path in controlnet_checkpoint_paths:
        control_net_inner = ControlNetModel.from_pretrained(checkpoint_path)
        controlnet_list.append(control_net_inner)
    original_controlnet_pipeline = (
        StableDiffusionControlNetPipeline.from_pretrained(
            STABLE_DIFFUSION_PATH,
            controlnet=controlnet_list,
            safety_checker=None,
            torch_dtype=DTYPE,
        ))
    original_controlnet_pipeline.scheduler = DDIMScheduler.from_config(original_controlnet_pipeline.scheduler.config)

    vae_control_1 = load_vae(control_type=CONTROL_TYPE_1)
    vae_control_2 = load_vae(control_type=CONTROL_TYPE_2)
    unet_control_1 = load_unet(control_type=CONTROL_TYPE_1)
    unet_control_2 = load_unet(control_type=CONTROL_TYPE_2)
    unet_unified = load_unet_unified(control_type_1=CONTROL_TYPE_1,
                                     control_type_2=CONTROL_TYPE_2,
                                     unet_control_1=unet_control_1,
                                     unet_control_2=unet_control_2)
    our_pipeline_2_controls = StableDiffusionTwoControlsPipeline.from_pretrained(
        STABLE_DIFFUSION_PATH,
        unet=unet_unified,
        vae_1=vae_control_1,
        vae_2=vae_control_2,
        safety_checker=None,
        torch_dtype=DTYPE
    )
    our_pipeline_2_controls.scheduler = DDIMScheduler.from_config(our_pipeline_2_controls.scheduler.config)

    # multi_controlnet = load_multi_controlnet(control_type_1="depth", control_type_2="hed")
    # our_controlnet_pipeline = (
    #     StableDiffusionControlNetPipeline.from_pretrained(
    #         STABLE_DIFFUSION_PATH,
    #         controlnet=multi_controlnet,
    #         safety_checker=None,
    #         torch_dtype=DTYPE,
    #     ))
    # our_controlnet_pipeline.scheduler = DDIMScheduler.from_config(our_controlnet_pipeline.scheduler.config)

    dataset = InterSemDataset(split="val",
                              dataset_to_use="COCO",
                              dataset_size=-1,
                              only_gt_image_and_prompt=(False if GEN_TYPE.ORIG_CONTROL in LIST_GEN_TYPES else True))
    tokenizer = AutoTokenizer.from_pretrained(STABLE_DIFFUSION_PATH,
                                              subfolder="tokenizer",
                                              use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(STABLE_DIFFUSION_PATH, subfolder="text_encoder")

    return (our_pipeline_2_controls,
            # our_controlnet_pipeline,
            original_controlnet_pipeline,
            stable_diffusion_pipeline,
            text_encoder,
            tokenizer,
            dataset)


def main():
    dist_url = f"file://{top_dir}/DDP/DDP_FILE_{np.random.randint(10000)}"
    # initialize_seed(1)
    torch.multiprocessing.set_sharing_strategy('file_system')
    world_size = torch.cuda.device_count()
    (our_pipeline_2_controls,
     # our_controlnet_pipeline,
     original_controlnet_pipeline,
     stable_diffusion_pipeline,
     text_encoder,
     tokenizer,
     dataset) = prepare_pipelines()
    if world_size > 1:
        mp.spawn(sample,
                 args=(
                     world_size,
                     our_pipeline_2_controls,
                     # our_controlnet_pipeline,
                     original_controlnet_pipeline,
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
               our_pipeline_2_controls=our_pipeline_2_controls,
               # our_controlnet_pipeline=our_controlnet_pipeline,
               original_controlnet_pipeline=original_controlnet_pipeline,
               stable_diffusion_pipeline=stable_diffusion_pipeline,
               text_encoder=text_encoder,
               tokenizer=tokenizer,
               dataset=dataset,
               dist_url=dist_url)


if __name__ == "__main__":
    main()
