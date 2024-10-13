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
    image_transforms_original_controlnet, remove_checkpoints_and_save_new_checkpoint
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, AutoConfig
from diffusers.pipelines import StableDiffusionPipeline
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from src.dataset import InterSemDataset, split_prompt
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
from diffusers.models.modeling_utils import ModelMixin
import lpips

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.22.0.dev0")

logger = get_logger(__name__)

negative_prompt = "lowres, bad anatomy, worst quality, low quality"


def run_validation_pipeline(vae,
                            args,
                            accelerator,
                            step,
                            val_dataloader):
    logger.info("Running validation... ")
    vae_scale_factor = 2 ** (len(accelerator.unwrap_model(vae).config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    list_images = []
    list_images_titles_validation = []
    curr_rec_loss_val = 0.0
    for idx, batch_val in enumerate(val_dataloader):
        image_logs = []
        validation_image_control = batch_val["control"][0]
        validation_prompt = batch_val["text"][0]
        images_transformed_val_control = torch.stack(
            [image_transforms_original_controlnet(orig_image, resolution=args.resolution)
             for orig_image in batch_val["control"]]).to(accelerator.device)
        generated_images_control = accelerator.unwrap_model(vae)(images_transformed_val_control).sample
        generated_images_control_post = \
            image_processor.postprocess(generated_images_control, output_type="pil", do_denormalize=[True])[0]
        image_logs.append({"validation_image_control": generated_images_control_post,
                           "images": [validation_image_control],
                           "validation_prompt": [validation_prompt]})
        text_for_plot = split_prompt(validation_prompt)
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                formatted_images = []
                for log in image_logs:
                    images = log["images"]
                    validation_prompt = log["validation_prompt"]
                    validation_image = log["validation_image"]
                    formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))
                    for image in images:
                        image = wandb.Image(image, caption=validation_prompt)
                        formatted_images.append(image)
                tracker.log({"validation": formatted_images})
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")
        list_images.extend([generated_images_control_post,
                            validation_image_control,
                            batch_val["pixel_values"][0]])
        list_images_titles_validation.extend([f"generated image control - {text_for_plot}",
                                              f"original image control - {text_for_plot}",
                                              f"ground truth image - {text_for_plot}"])
        # predicted_images_transformed_val = torch.stack(
        #     [image_transforms_original_controlnet(image, resolution=args.resolution)
        #      for image in [generated_images]]).to(accelerator.device)
        # latents_val = (vae_unwrapped.encode(images_transformed_val.to(accelerator.device)).latent_dist.sample())
        # latents_val = latents_val * vae_unwrapped.config.scaling_factor
        curr_rec_loss_val += F.mse_loss(images_transformed_val_control.float(),
                                        generated_images_control.float(),
                                        reduction="mean").item()
        plot_images_and_prompt(list_images,
                               list_images_titles_validation,
                               f"validation-step-{step}")
    return curr_rec_loss_val / len(val_dataloader)


def collate_fn(examples):
    return {
        "pixel_values": [example["jpg"] for example in examples],
        "control": [example[f"{args.control_type}"] for example in examples],
        "text": [example["text"] for example in examples],
    }


def initialize_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_last_checkpoint_dir(output_dir):
    dirs = os.listdir(output_dir)
    dirs = [d for d in dirs if d.startswith(f"{args.checkpoint_prefix}")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))
    path = dirs[-1] if len(dirs) > 0 else None
    return path


def main(args, trial=None):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
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

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    if isinstance(model, AutoencoderKL):
                        torch.save(model.state_dict(), os.path.join(output_dir, f"vae.pt"))
                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                if isinstance(model, AutoencoderKL):
                    model.load_state_dict(torch.load(os.path.join(input_dir, f"vae.pt")))

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # if args.gradient_checkpointing:
    #     vae.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(vae).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(vae).dtype}. {low_precision_error_string}"
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
        train_dataset = InterSemDataset(split="validation", dataset_to_use="COCO",
                                        dataset_size=args.dataset_size, sampling_mode=True)
    else:
        train_dataset = InterSemDataset(split="train", dataset_to_use="COCO",
                                        dataset_size=args.dataset_size, sampling_mode=True)
    val_dataset = InterSemDataset(split="validation", dataset_to_use="COCO",
                                  dataset_size=1, sampling_mode=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples=examples),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples=examples),
        batch_size=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    vae.train()

    # Optimizer creation
    params_to_optimize = vae.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        # num_cycles=args.lr_num_cycles,
        # power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(vae, optimizer, train_dataloader, lr_scheduler)

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

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            path = get_last_checkpoint_dir(args.output_dir)
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
            else:
                best_loss = torch.load(f"{load_loss_path}/loss.pt")["best_loss"]
            logger.info(f"loaded checkpoint with loss: {best_loss}")
            global_step = int(path.split("-")[-1])

            with open(f"{os.path.join(args.output_dir, path)}/hyper_params_config.json") as json_file:
                saved_config = json.load(json_file)
                if saved_config["learning_rate"] != args.learning_rate:
                    load_optimizer = False

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # if not load_optimizer:
    #     for ind in range(len(optimizer.param_groups)):
    #         optimizer.param_groups[ind].update({'lr': args.learning_rate})
    #     for optim in lr_scheduler.optimizers:
    #         for ind in range(len(optim.param_groups)):
    #             optim.param_groups[ind].update({'lr': args.learning_rate})

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

    lpips_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device)
    lpips_loss_fn.requires_grad_(False)

    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"epoch num: {epoch}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(vae):
                # Convert images to latent space
                control_images_transformed = torch.stack([image_transforms_original_controlnet(
                    control_image.convert("RGB"), resolution=args.resolution).to(accelerator.device)
                                                          for control_image in batch["control"]])
                model_pred = vae(control_images_transformed).sample
                mse_loss = F.mse_loss(model_pred.float(), control_images_transformed.float(), reduction="mean")
                with torch.no_grad():
                    lpips_loss = lpips_loss_fn(model_pred.float(), control_images_transformed.float()).mean()
                    if not torch.isfinite(lpips_loss):
                        lpips_loss = torch.tensor(0)
                loss = (mse_loss + 5e-1 * lpips_loss)
                if not torch.isfinite(loss):
                    logger.info("WARNING: non-finite loss")

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
                        vae.eval()
                        with torch.no_grad():
                            curr_loss = run_validation_pipeline(
                                vae=vae,
                                args=args,
                                accelerator=accelerator,
                                step=global_step,
                                val_dataloader=val_dataloader
                            )
                            losses_dict_training_all_steps[f"rec_loss"].append(np.array(losses_dict_single_step[
                                                                                            f"rec_loss"]).mean().item())
                            for loss_key in losses_dict_single_step:
                                losses_dict_single_step[loss_key] = []
                            losses_dict_validation_all_steps[f"rec_loss"].append(curr_loss)
                        vae.train()

                        plt.clf()
                        plt.figure(figsize=(40, 20))
                        plt.plot(losses_dict_validation_all_steps[f"rec_loss"],
                                 label=f"validation - reconstruction loss")
                        plt.legend(loc="upper left")
                        plt.tight_layout()
                        plt.savefig(f"{top_dir}/InterSem/test_imgs/loss.png")
                        plt.close()

                        # if accelerator.is_main_process:
                        # if global_step % args.checkpointing_steps == 0:
                        if not args.overfit_on_single_image:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
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
    args = parse_args()
    args.output_dir = f"{os.path.dirname(os.path.realpath(__file__))}"
    args.checkpoint_prefix = f"vae-{args.control_type}-checkpoint"
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
