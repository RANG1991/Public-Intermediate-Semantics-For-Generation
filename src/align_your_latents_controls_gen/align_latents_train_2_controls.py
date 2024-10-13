#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

top_dir = Path(__file__).resolve().parent.parent.parent.parent.absolute()

sys.path.append(str(top_dir / "InterSem"))

# define_locations_for_hugging_face()

from src.utils.project_utils import plot_images_and_prompt
import accelerate
import shutil
import transformers
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from src.utils.project_utils import conditioning_image_transforms_original_controlnet, \
    image_transforms_original_controlnet, get_last_checkpoint_dir, compute_FID_metrics, \
    remove_checkpoints_and_save_new_checkpoint, split_prompt
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from src.align_your_latents_controls_gen.align_latents_pipeline_2_controls import StableDiffusionTwoControlsPipeline
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UniPCMultistepScheduler,
    UNet2DConditionModel,
    ControlNetModel,
    StableDiffusionControlNetPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from src.dataset import InterSemDataset
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import json
from ruamel.yaml import YAML
from collections import OrderedDict
import random
from scipy.stats import bernoulli
from accelerate import DistributedDataParallelKwargs
import optuna
from optuna.trial import TrialState
from functools import partial
import itertools
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.image_processor import VaeImageProcessor
from src.align_your_latents_controls_gen.unified_unet import UnetConditionModelUnified
from src.align_your_latents_controls_gen.cross_attn_unified_unet import UnetConditionCrossAttnUnified
import re
from src.align_your_latents_controls_gen.align_latents_train_control_gen import negative_prompt, \
    dict_control_type_to_pref
import src.utils.upload_to_hugging_face_hub
from huggingface_hub import HfApi, login

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.22.0.dev0")

logger = get_logger(__name__)


def run_validation_pipeline(vaes_list,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            step,
                            val_dataloader,
                            dataset):
    logger.info("Running validation... ")

    pipeline_validation_two_controls = StableDiffusionTwoControlsPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        safety_checker=None,
        unet=accelerator.unwrap_model(unet),
        vae_1=vaes_list[0],
        vae_2=vaes_list[1],
        # revision=args.revision,
        # variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline_validation_two_controls.scheduler = DDIMScheduler.from_config(
        pipeline_validation_two_controls.scheduler.config)
    pipeline_validation_two_controls = pipeline_validation_two_controls.to(accelerator.device)
    pipeline_validation_two_controls.set_progress_bar_config(leave=True)
    pipeline_validation_two_controls.set_progress_bar_config(position=0)

    controlnet_list = []
    for checkpoint_path in args.controlnets_path:
        control_net_inner = ControlNetModel.from_pretrained(checkpoint_path)
        controlnet_list.append(control_net_inner)
    original_controlnet_pipeline = (
        StableDiffusionControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            controlnet=controlnet_list,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )).to(accelerator.device)
    original_controlnet_pipeline.scheduler = DDIMScheduler.from_config(original_controlnet_pipeline.scheduler.config)
    original_controlnet_pipeline.set_progress_bar_config(leave=True)
    original_controlnet_pipeline.set_progress_bar_config(position=0)

    list_ground_truth_controls_1 = []
    list_ground_truth_controls_2 = []
    list_ground_truth_images = []
    list_prompts_1 = []
    list_prompts_2 = []
    list_prompts_unified = []
    list_prompt_embeds_unified = []
    list_prompt_embeds_neg_unified = []
    for idx, batch_val in enumerate(val_dataloader):
        list_prompts_1.extend(batch_val["text"])
        list_prompts_2.extend(batch_val["text"])
        list_prompts_unified.extend(batch_val["text"])
        list_ground_truth_controls_1.extend(batch_val["control_1"])
        list_ground_truth_controls_2.extend(batch_val["control_2"])
        list_ground_truth_images.extend(batch_val["pixel_values"])
        list_prompt_embeds_unified.append(batch_val["txt_embeddings_unet_unified"])
        list_prompt_embeds_neg_unified.append(batch_val["txt_embeddings_negative"])
    list_prompts_1 = [dict_control_type_to_pref[args.control_type_1] + validation_prompt.lower() for validation_prompt
                      in list_prompts_1]
    list_prompts_2 = [dict_control_type_to_pref[args.control_type_2] + validation_prompt.lower() for validation_prompt
                      in list_prompts_2]
    # list_prompts_unified = [validation_prompt.lower() for validation_prompt in list_prompts_unified]
    list_neg_prompts = [negative_prompt] * len(list_prompts_1)
    list_images = []
    image_logs = []
    list_images_titles_validation = []
    list_generated_images = []
    torch.cuda.empty_cache()
    num_image_per_run = 1
    curr_loss_val = 0.0
    for j in range(math.ceil(len(list_prompts_1) / num_image_per_run)):
        list_prompts_1_curr = list_prompts_1[j * num_image_per_run: (j + 1) * num_image_per_run]
        list_prompts_2_curr = list_prompts_2[j * num_image_per_run: (j + 1) * num_image_per_run]
        list_prompts_unified_curr = list_prompts_unified[j * num_image_per_run: (j + 1) * num_image_per_run]
        list_ground_truth_images_curr = list_ground_truth_images[j * num_image_per_run: (j + 1) * num_image_per_run]
        list_prompt_embeds_unified_curr = list_prompt_embeds_unified[j * num_image_per_run: (j + 1) * num_image_per_run]
        list_prompt_embeds_neg_unified_curr = list_prompt_embeds_neg_unified[
                                              j * num_image_per_run: (j + 1) * num_image_per_run]
        prompts_input = [list(prompts_input) for prompts_input in
                         zip(list_prompts_1_curr, list_prompts_2_curr, list_prompts_unified_curr)][0]

        with torch.autocast("cuda"):
            generator = torch.Generator(device=accelerator.device).manual_seed(1)
            pipeline_output = pipeline_validation_two_controls(
                prompts=prompts_input,
                negative_prompt=negative_prompt,
                num_inference_steps=70,
                generator=generator,
                width=args.resolution,
                height=args.resolution,
            )
            list_generated_controls_1_curr = [generated_control for generated_control in pipeline_output.control_1]
            list_generated_controls_2_curr = [generated_control for generated_control in pipeline_output.control_2]

            images_input_original_controlnet = [list(images_input_tuple) for images_input_tuple in
                                                zip(list_generated_controls_1_curr, list_generated_controls_2_curr)]

            # for controls_pair_ind in range(len(images_input_original_controlnet)):
            #     plot_images_and_prompt(images_input_original_controlnet[controls_pair_ind],
            #                            ["", ""],
            #                            prompts_input[2])

            prompts_embeds_input_original_controlnet = torch.cat(list_prompt_embeds_unified_curr).to(accelerator.device)
            prompts_embeds_neg_input_original_controlnet = torch.cat(list_prompt_embeds_neg_unified_curr).to(
                accelerator.device)

            # print(len(images_input_original_controlnet))
            # print(prompts_embeds_input_original_controlnet.shape)
            # print(prompts_embeds_neg_input_original_controlnet.shape)

            generated_image_control, _ = original_controlnet_pipeline(
                image=images_input_original_controlnet,
                prompt_embeds=prompts_embeds_input_original_controlnet,
                negative_prompt_embeds=prompts_embeds_neg_input_original_controlnet,
                # controlnet_conditioning_scale=[0.5, 0.5],
                num_inference_steps=70,
                # guidance_scale=14,
                generator=generator,
                width=args.resolution,
                hieght=args.resolution,
                return_dict=False,
            )
        list_generated_images_curr = [generated_image for generated_image in generated_image_control]
        list_generated_images.extend(list_generated_images_curr)
        image_logs.append({"validation_image": list_ground_truth_images_curr,
                           "images": list_generated_images_curr,
                           "validation_prompt": list_prompts_unified_curr})
        for i, curr_prompt in enumerate(list_prompts_unified_curr):
            text_for_plot = split_prompt(curr_prompt)
            list_images.extend([list_generated_images_curr[i],
                                list_generated_controls_1_curr[i],
                                list_generated_controls_2_curr[i],
                                list_ground_truth_images_curr[i]])
            list_images_titles_validation.extend([f"generated image - {text_for_plot}",
                                                  f"generated control 1 - {text_for_plot}",
                                                  f"generated control 2 - {text_for_plot}",
                                                  f"ground truth image - {text_for_plot}"])

            # curr_loss_val += ((np.asarray(dataset.get_hed(list_generated_images_curr[i])) -
            #                    np.asarray(list_generated_controls_2_curr[i])) ** 2).mean()

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                for image in validation_image:
                    formatted_images.append(wandb.Image(image, caption="Controlnet conditioning"))
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)
            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    curr_loss_val = compute_FID_metrics(real_images=list_ground_truth_images,
                                        fake_images=list_generated_images,
                                        resolution=args.resolution,
                                        device=accelerator.device)

    plot_images_and_prompt(list_images,
                           list_images_titles_validation,
                           f"validation-step-{step}")
    print(f"FID is: {curr_loss_val}")
    return curr_loss_val


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def collate_fn(examples, tokenizer, text_encoder, device, zero_out_prob, tokenizer_max_length,
               control_type_1, control_type_2):
    txt_embeddings_unet_control_1 = []
    txt_embeddings_unet_control_2 = []
    txt_embeddings_unet_unified = []
    txt_negative_embeddings = []
    attention_masks = []
    inputs_ids = []
    input_negative_prompt = tokenizer(negative_prompt,
                                      max_length=tokenizer_max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt').to(device)
    input_empty_prompt = tokenizer("",
                                   max_length=tokenizer_max_length,
                                   padding='max_length',
                                   truncation=True,
                                   return_tensors='pt').to(device)
    default_prompt = tokenizer("a high-quality and extremely detailed image",
                               max_length=tokenizer_max_length,
                               padding='max_length',
                               truncation=True,
                               return_tensors='pt').to(device)
    for example in examples:
        with torch.no_grad():
            (zero_out_unet_control_1_text_embeddings,
             zero_out_unet_text_embeddings) = bernoulli.rvs(size=2, p=zero_out_prob)
            tokenized_example_control_1 = tokenizer(
                dict_control_type_to_pref[control_type_1] + example["text"].lower(),
                max_length=tokenizer_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt').to(device)
            tokenized_example_control_2 = tokenizer(
                dict_control_type_to_pref[control_type_2] + example["text"].lower(),
                max_length=tokenizer_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt').to(device)
            tokenized_example_unified = tokenizer(
                example["text"],
                max_length=tokenizer_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt').to(device)
            # prompt encoding
            embeddings_text_encoder_control_1 = text_encoder(input_ids=tokenized_example_control_1["input_ids"],
                                                             attention_mask=tokenized_example_control_1[
                                                                 "attention_mask"])[0]
            # prompt encoding
            embeddings_text_encoder_control_2 = text_encoder(input_ids=tokenized_example_control_2["input_ids"],
                                                             attention_mask=tokenized_example_control_2[
                                                                 "attention_mask"])[0]
            # prompt encoding
            embeddings_text_encoder_unified = text_encoder(input_ids=tokenized_example_unified["input_ids"],
                                                           attention_mask=tokenized_example_unified["attention_mask"])[
                0]
            # empty prompt encoding
            embeddings_text_encoder_empty_string = text_encoder(**input_empty_prompt)[0]
            if zero_out_unet_text_embeddings:
                txt_embeddings_unet_control_1.append(embeddings_text_encoder_empty_string.squeeze(0))
                txt_embeddings_unet_control_2.append(embeddings_text_encoder_empty_string.squeeze(0))
                txt_embeddings_unet_unified.append(embeddings_text_encoder_empty_string.squeeze(0))
            else:
                txt_embeddings_unet_control_1.append(embeddings_text_encoder_control_1.squeeze(0))
                txt_embeddings_unet_control_2.append(embeddings_text_encoder_control_2.squeeze(0))
                txt_embeddings_unet_unified.append(embeddings_text_encoder_unified.squeeze(0))
            # negative prompt
            embeddings_negative_original_encoder_seq = text_encoder(**input_negative_prompt)[0]
            txt_negative_embeddings.append(embeddings_negative_original_encoder_seq.squeeze(0))
    return {
        "pixel_values": [example["jpg"] for example in examples],
        "txt_embeddings_unet_control_1": torch.stack(txt_embeddings_unet_control_1),
        "txt_embeddings_unet_control_2": torch.stack(txt_embeddings_unet_control_2),
        "txt_embeddings_unet_unified": torch.stack(txt_embeddings_unet_unified),
        "control_1": [example[f"{control_type_1}"] for example in examples],
        "control_2": [example[f"{control_type_2}"] for example in examples],
        "text": [example["text"] for example in examples],
        "txt_embeddings_negative": torch.stack(txt_negative_embeddings),
    }


def initialize_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def remove_noise(
        noisy_samples,
        noise,
        timesteps,
        alphas_cumprod):
    timesteps = timesteps.to(noisy_samples.device)
    alphas_cumprod = alphas_cumprod.to(device=noisy_samples.device)
    alphas_cumprod = alphas_cumprod.to(dtype=noisy_samples.dtype)
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(noisy_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    original_samples = (noisy_samples - (sqrt_one_minus_alpha_prod * noise)) / sqrt_alpha_prod
    return original_samples


def generate_image_from_noisy_latent(vae,
                                     scaling_factor,
                                     noisy_samples,
                                     noise,
                                     timesteps,
                                     alphas_cumprod):
    decoded_image = vae.decode(remove_noise(noisy_samples, noise, timesteps, alphas_cumprod) / scaling_factor).sample
    image = decoded_image[0].unsqueeze(0)
    # image = vae_image_processor.postprocess(image, output_type="pil", do_denormalize=[True])
    # image[0].save(f"{top_dir}/InterSem/test_imgs/check.png")
    return image


def main(args, trial=None):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(
            broadcast_buffers=False,
            # find_unused_parameters=True,
        )]
    )

    # For mixed precision training, we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    # Load scheduler and models
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path,
                                                    subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path,
                                                    subfolder="text_encoder",
                                                    revision=args.revision,
                                                    variant=args.variant)

    vae_control_1 = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                                  subfolder="vae",
                                                  revision=args.revision,
                                                  variant=args.variant)
    checkpoint = get_last_checkpoint_dir(output_dir=args.output_dir,
                                         prefix=f"vae-{args.control_type_1}-checkpoint",
                                         accelerator=accelerator)
    vae_control_1.load_state_dict(torch.load(f"{checkpoint}/vae.pt"), strict=True)

    vae_control_2 = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                                  subfolder="vae",
                                                  revision=args.revision,
                                                  variant=args.variant)
    checkpoint = get_last_checkpoint_dir(output_dir=args.output_dir,
                                         prefix=f"vae-{args.control_type_2}-checkpoint",
                                         accelerator=accelerator)
    vae_control_2.load_state_dict(torch.load(f"{checkpoint}/vae.pt"), strict=True)

    unet_control_1 = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,
                                                          subfolder="unet",
                                                          revision=args.revision,
                                                          variant=args.variant)
    checkpoint = get_last_checkpoint_dir(output_dir=args.output_dir,
                                         prefix=f"control-generation-checkpoint-{args.control_type_1}",
                                         get_best=True,
                                         accelerator=accelerator)
    unet_control_1.load_state_dict(torch.load(f"{checkpoint}/unet.pt"), strict=True)

    unet_control_2 = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,
                                                          subfolder="unet",
                                                          revision=args.revision,
                                                          variant=args.variant)
    checkpoint = get_last_checkpoint_dir(output_dir=args.output_dir,
                                         prefix=f"control-generation-checkpoint-{args.control_type_2}",
                                         get_best=True,
                                         accelerator=accelerator)
    unet_control_2.load_state_dict(torch.load(f"{checkpoint}/unet.pt"), strict=True)

    unet = UnetConditionModelUnified(unets_list=[unet_control_1, unet_control_2])

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    if isinstance(model, UnetConditionModelUnified):
                        torch.save(model.state_dict(),
                                   os.path.join(output_dir, f"unet.pt"))
                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                if isinstance(model, UnetConditionModelUnified):
                    model.load_state_dict(
                        torch.load(os.path.join(input_dir, f"unet.pt")))

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    text_encoder.requires_grad_(False)
    vae_control_1.requires_grad_(False)
    vae_control_2.requires_grad_(False)

    # # vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_control_1.config.scaling_factor)

    # if args.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()
    #     vae.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if args.overfit_on_single_image:
        train_dataset = InterSemDataset(split="validation", dataset_to_use="COCO", dataset_size=args.dataset_size)
    else:
        train_dataset = InterSemDataset(split="train", dataset_to_use="COCO", dataset_size=args.dataset_size)
    val_dataset = InterSemDataset(split="validation", dataset_to_use="COCO", dataset_size=20)
    # val_dataset = torch.utils.data.Subset(val_dataset, list(range(19, 20)))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples=examples,
                                               device=accelerator.device,
                                               tokenizer=tokenizer,
                                               text_encoder=text_encoder,
                                               zero_out_prob=args.probability_zero_out_input,
                                               tokenizer_max_length=args.tokenizer_max_length,
                                               control_type_1=args.control_type_1,
                                               control_type_2=args.control_type_2,
                                               ),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples=examples,
                                               device=accelerator.device,
                                               tokenizer=tokenizer,
                                               text_encoder=text_encoder,
                                               zero_out_prob=0.0,
                                               tokenizer_max_length=args.tokenizer_max_length,
                                               control_type_1=args.control_type_1,
                                               control_type_2=args.control_type_2,
                                               ),
        batch_size=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    unet.train()
    # vae.train()

    # Optimizer creation
    params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=(((args.lr_warmup_steps // 1) + 1)
                          * accelerator.num_processes),
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        # num_cycles=args.lr_num_cycles,
        # power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader,
                                                                          lr_scheduler)

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae_control_1.to(accelerator.device, dtype=weight_dtype)
    vae_control_2.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    vaes_list = [vae_control_1, vae_control_2]

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("memory: ", torch.cuda.mem_get_info()[1] / (10 ** 9))

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    best_loss = None
    curr_loss = None
    load_optimizer = True

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            path = get_last_checkpoint_dir(output_dir=args.output_dir, prefix=args.checkpoint_prefix,
                                           accelerator=accelerator)
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            load_loss_path = os.path.join(args.output_dir, path)
            if Path(f"{load_loss_path}/loss.json").exists():
                with open(f"{load_loss_path}/loss.json") as f:
                    best_loss = json.load(f)["best_loss"]
            logger.info(f"loaded checkpoint with loss: {best_loss}")
            global_step = int(path.replace("-best", "").split("-")[-1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

            with open(f"{os.path.join(args.output_dir, path)}/hyper_params_config.json") as json_file:
                saved_config = json.load(json_file)
                if saved_config["learning_rate"] != args.learning_rate:
                    print(f"learning rate ({args.learning_rate}) is not equal to the learning rate of the saved model "
                          f"({saved_config['learning_rate']}) - changing the learning rate to the new learning rate")
                    load_optimizer = False

    else:
        initial_global_step = 0

    if not load_optimizer:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=(((args.lr_warmup_steps // 1) + 1)
                              * accelerator.num_processes),
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            # num_cycles=args.lr_num_cycles,
            # power=args.lr_power,
        )
        lr_scheduler = accelerator.prepare(lr_scheduler)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    losses_dict_training_all_steps = {f"rec_loss": []}
    losses_dict_validation_all_steps = {f"rec_loss": []}
    losses_dict_single_step = {f"rec_loss": []}

    # if accelerator.is_main_process:
    #     with torch.no_grad():
    #         unet.eval()
    #         run_validation_pipeline(
    #             vaes_list=vaes_list,
    #             unet=accelerator.unwrap_model(unet),
    #             args=args,
    #             accelerator=accelerator,
    #             weight_dtype=weight_dtype,
    #             step=global_step,
    #             val_dataloader=val_dataloader
    #         )
    #         unet.train()

    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"epoch num: {epoch}")
        for step, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            with accelerator.accumulate(unet):
                # Convert images to latent space
                samples_list = []
                timesteps_list = []
                encoder_hidden_states_list = []
                noises_list = []
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (args.train_batch_size,),
                                          device=accelerator.device)
                timesteps = timesteps.long()
                for i in range(2):
                    control_images_transformed = torch.stack([image_transforms_original_controlnet(
                        cond_image.convert("RGB"), resolution=args.resolution).to(accelerator.device).to(weight_dtype)
                                                              for cond_image in batch[f"control_{i + 1}"]])
                    latents = vaes_list[i].encode(control_images_transformed).latent_dist.sample()
                    latents = latents * vaes_list[i].config.scaling_factor
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    samples_list.append(noisy_latents)
                    timesteps_list.append(timesteps)
                    encoder_hidden_states_list.append(
                        batch[f"txt_embeddings_unet_control_{i + 1}"].to(dtype=weight_dtype))
                    noises_list.append(noise)
                # Predict the noise residual
                model_pred = unet(
                    samples_list=samples_list,
                    timesteps_list=timesteps_list,
                    encoder_hidden_states_list=encoder_hidden_states_list,
                    encoder_hidden_states_unified=batch[f"txt_embeddings_unet_unified"].to(dtype=weight_dtype),
                ).sample

                # controls_list = []
                # for i in range(2):
                #     scaling_factor = vaes_list[i].config.scaling_factor
                #     generated_control = generate_image_from_noisy_latent(vae=vaes_list[i],
                #                                                          scaling_factor=scaling_factor,
                #                                                          noisy_samples=samples_list[i],
                #                                                          noise=model_pred[i].to(weight_dtype),
                #                                                          timesteps=timesteps_list[i],
                #                                                          alphas_cumprod=noise_scheduler.alphas_cumprod)
                #     controls_list.append(generated_control.to(torch.float32))

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                reconstruction_loss_unet = 0.0

                for i in range(len(model_pred)):
                    # print(torch.unique(model_pred[i]))
                    # print(torch.unique(noises_list[i]))
                    reconstruction_loss_unet += F.mse_loss(model_pred[i].float(), noises_list[i].float(),
                                                           reduction="mean")
                    # print(reconstruction_loss_unet)

                reconstruction_loss_unet_all = (1 / (len(model_pred))) * reconstruction_loss_unet

                # latents_generated_image, _ = original_controlnet_pipeline(
                #     image=controls_list,
                #     negative_prompt=[negative_prompt] * len(batch["text"]),
                #     prompt=batch["text"],
                #     num_inference_steps=50,
                #     # guidance_scale=10.0,
                #     # generator=generator,
                #     width=args.resolution,
                #     height=args.resolution,
                #     output_type="latent",
                #     return_dict=False,
                # )

                # original_images_transformed = torch.stack([image_transforms_original_controlnet(
                #     image.convert("RGB"), resolution=args.resolution).to(accelerator.device).to(weight_dtype)
                #                                            for image in batch["pixel_values"]])
                #
                # latents_original_image = vaes_list[0].encode(original_images_transformed).latent_dist.sample()
                # loss = F.mse_loss(latents_generated_image.float(), latents_original_image.float())

                loss = reconstruction_loss_unet_all

                # curr_loss = loss.detach().item()
                losses_dict_single_step[f"rec_loss"].append(loss.detach().item())

                if trial is not None:
                    trial.report(loss.detach().item(), global_step)
                if trial is not None:
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = params_to_optimize
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.validation_steps == 0:
                        unet.eval()
                        with torch.no_grad():
                            curr_loss = run_validation_pipeline(
                                vaes_list=vaes_list,
                                unet=unet,
                                args=args,
                                accelerator=accelerator,
                                weight_dtype=weight_dtype,
                                step=global_step,
                                val_dataloader=val_dataloader,
                                dataset=val_dataset,
                            )
                            losses_dict_training_all_steps[f"rec_loss"].append(np.array(losses_dict_single_step[
                                                                                            f"rec_loss"]).mean().item())
                            for loss_key in losses_dict_single_step:
                                losses_dict_single_step[loss_key] = []
                            losses_dict_validation_all_steps[f"rec_loss"].append(curr_loss)
                        unet.train()

                        plt.clf()
                        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 20))
                        ax1.plot(losses_dict_training_all_steps[f"rec_loss"],
                                 label=f"training - reconstruction loss")
                        ax1.legend(loc="upper left")
                        ax1.set_title('training - reconstruction loss')
                        ax2.plot(losses_dict_validation_all_steps[f"rec_loss"],
                                 label=f"validation - reconstruction loss")
                        ax2.legend(loc="upper left")
                        ax2.set_title('validation - reconstruction loss')
                        plt.tight_layout()
                        plt.savefig(f"{top_dir}/InterSem/test_imgs/loss.png")
                        plt.close()

                        # if accelerator.is_main_process:
                        # if global_step % args.checkpointing_steps == 0:
                        if not args.overfit_on_single_image:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            best_loss = remove_checkpoints_and_save_new_checkpoint(args, logger, accelerator,
                                                                                   global_step,
                                                                                   best_loss, curr_loss)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    return np.array(losses_dict_validation_all_steps[f"rec_loss"]).mean().item()


class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config_file_name',
                        help='file name of the configuration (in yaml file format)',
                        required=True)
    command_args = parser.parse_args()
    yaml_path = Path(command_args.yaml_config_file_name).resolve()
    if yaml_path.exists():
        with yaml_path.open('r') as fp_read:
            yaml = YAML(typ="safe")
            args = Map(yaml.load(fp_read))
        # with yaml_path.open('w') as fp_write:
        #     yaml = YAML(typ="safe")
        #     yaml.indent()
        #     yaml.dump(dict(OrderedDict(sorted(args.items()))), fp_write)
    else:
        raise FileNotFoundError(yaml_path)
    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the "
            "controlnet encoder.")
    return args


def prepare_hyperparameters_for_search(args, trial):
    args["num_train_epochs"] = 2
    args["max_train_steps"] = 30
    args["train_batch_size"] = 2
    args["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1e-5, log=True)
    args["gradient_accumulation_steps"] = trial.suggest_int("gradient_accumulation_steps", 1, 1, log=True)
    args["adam_beta1"] = trial.suggest_float("adam_beta1", 0.9, 0.99, log=True)
    args["adam_beta2"] = trial.suggest_float("adam_beta2", 0.99, 0.999, log=True)
    args["adam_weight_decay"] = trial.suggest_float("adam_weight_decay", 1e-4, 1e-1, log=True)
    args["adam_epsilon"] = trial.suggest_float("adam_epsilon", 1e-09, 1e-08, log=True)
    args["max_grad_norm"] = trial.suggest_int("max_grad_norm", 1, 4, log=True)
    return args


def objective(trial, args):
    args = prepare_hyperparameters_for_search(args, trial)
    args_sorted_for_print = dict(OrderedDict(sorted(vars(args).items())))
    print(json.dumps(args_sorted_for_print, indent=4))
    reconstruction_loss = main(args, trial)
    return reconstruction_loss


if __name__ == "__main__":
    # initialize_seed(1)
    torch.cuda.empty_cache()
    args = parse_args()
    # if args.control_type not in dict_control_type_to_pref:
    #     raise Exception(f"The control type argument: {args.control_type} is not in the control types dictionary")
    args.output_dir = f"{os.path.dirname(os.path.realpath(__file__))}"
    args.checkpoint_prefix = f"control-generation-checkpoint-{args.control_type_1}-{args.control_type_2}"
    args.controlnets_path = [
        (f"thibaud/controlnet-sd21-{args.control_type_1}-diffusers" if args.control_type_1 in ["hed", "depth"]
         else f"thibaud/controlnet-sd21-ade20k-diffusers" if args.control_type_1 == "seg" else None),
        (f"thibaud/controlnet-sd21-{args.control_type_2}-diffusers" if args.control_type_2 in ["hed", "depth"]
         else f"thibaud/controlnet-sd21-ade20k-diffusers" if args.control_type_2 == "seg" else None)]
    if args.run_optuna:
        study = optuna.create_study(direction="minimize")
        objective = partial(objective, args=args)
        study.optimize(objective, n_trials=100, timeout=None)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(" {}: {}".format(key, value))
    else:
        args_sorted_for_print = dict(OrderedDict(sorted(vars(args).items())))
        print(json.dumps(args_sorted_for_print, indent=4))
        main(args)
