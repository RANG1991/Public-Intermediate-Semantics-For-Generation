import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
top_dir = Path(__file__).parent.parent.parent.parent

import cv2
import imageio
import numpy as np
from datasets import DatasetDict
from matplotlib import pyplot as plt
import math
from pathlib import Path
from PIL import Image
from src.utils.calculate_FID import calculate_frechet_distance
import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import re
from huggingface_hub import HfApi, login, HfFileSystem
from huggingface_hub.utils import EntryNotFoundError
import shutil
from src.utils.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.utils import save_image
import random
from collections import OrderedDict
import json
from contextlib import nullcontext


def split_prompt(prompt, num_splits=4):
    splits = []
    prompt_split = prompt.split(" ")
    for num_split in range(num_splits):
        splits.append(" ".join(prompt_split[num_split * math.ceil(len(prompt_split) / num_splits):
                                            (num_split + 1) * math.ceil(len(prompt_split) / num_splits)]))
    new_prompt = "\n".join(splits)
    return new_prompt


def create_image_table(d_prompt_and_type_to_image, text_prompts, control_types, two_controls=False):
    height_ratios = (len(text_prompts) + 1) * [1]
    height_ratios[0] = 0.1
    width_ratios = (len(control_types) + 1) * [2]
    if two_controls:
        width_ratios[0] = 0.5
        figsize = (27, 18)
    else:
        width_ratios[-1] = 1
        width_ratios[0] = 0.8
        figsize = (20, 15)
    fig, ax = plt.subplots(len(text_prompts) + 1, len(control_types) + 1, figsize=figsize,
                           gridspec_kw={'width_ratios': width_ratios, "height_ratios": height_ratios})
    for i in range(len(text_prompts) + 1):
        for j in range(len(control_types) + 1):
            curr_ax = ax[i][j]
            curr_ax.axis('off')
            if j == 0 and i > 0:
                curr_ax.text(0.5,
                             0.5,
                             split_prompt(text_prompts[i - 1], num_splits=6),
                             size="18",
                             weight='bold',
                             horizontalalignment='center',
                             verticalalignment='center')
            elif i == 0 and j > 0:
                curr_ax.text(0.5,
                             0.5,
                             control_types[j - 1],
                             size="18",
                             weight='bold',
                             horizontalalignment='center',
                             verticalalignment='center')
            elif i > 0 and j > 0:
                image_to_plot = d_prompt_and_type_to_image[(text_prompts[i - 1], control_types[j - 1])]
                curr_ax.imshow(image_to_plot)
    fig.tight_layout()
    if two_controls:
        fig.subplots_adjust(wspace=0.01, hspace=0)
    else:
        fig.subplots_adjust(wspace=0.01, hspace=0)
    fig.show()
    if two_controls:
        fig.savefig(r"C:\Users\galun\Desktop\Results_Figure_2_controls.png")
    else:
        fig.savefig(r"C:\Users\galun\Desktop\Results_Figure_separate_controls.png")


def generate_collage_from_good_or_bad_images(folder_path):
    if "2_controls" in folder_path:
        two_controls = True
    else:
        two_controls = False
    file_paths = [folder_path / Path(f) for f in os.listdir(path=folder_path) if
                  os.path.isfile(os.path.join(folder_path, f))]
    d_prompt_and_type_to_image = {}
    text_prompts = set()
    control_types = set()
    for file_path in file_paths:
        text_prompt = re.sub('^(.*)_text_control_gen_', '', file_path.stem).replace(".", "")
        control_type = re.sub('_text_control_gen_(.*)$', '', file_path.stem).replace(".", "")
        if control_type == "SD":
            control_type = "Without control"
        if control_type == "seg":
            control_type = "Segmentation map"
        if control_type == "depth":
            control_type = "Depth map"
        if control_type == "hed":
            control_type = "Hough lines"
        if control_type == "depth_seg":
            control_type = "Depth map and Segmentation map"
        if control_type == "depth_hed":
            control_type = "Depth map and Hough lines"
        text_prompts.add(text_prompt)
        control_types.add(control_type)
        d_prompt_and_type_to_image[(text_prompt, control_type)] = Image.open(file_path)
    control_types = sorted(control_types)
    create_image_table(d_prompt_and_type_to_image, list(text_prompts), list(control_types), two_controls=two_controls)


def zero_module(module):
    for p in module.parameters():
        torch.nn.init.zeros_(p)
    return module


def initialize_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_lowest_loss_checkpoint_dir(output_dir, sub_dir, checkpoint_prefix):
    dirs = os.listdir(output_dir)
    dirs = [d for d in dirs if re.search(rf"{checkpoint_prefix}-[0-9]+", d) is not None]

    def get_loss(dir):
        with open(f"{dir}/{sub_dir}/loss.json") as f:
            best_loss = json.load(f)["best_loss"]
        return best_loss

    dirs = sorted(dirs, key=get_loss)
    path = dirs[0] if len(dirs) > 0 else None
    return path


def remove_checkpoints_and_save_new_checkpoint(args, logger, accelerator, global_step, best_loss, curr_loss):
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(args.output_dir)
        checkpoints = [d for d in checkpoints if
                       re.search(rf"{args.checkpoint_prefix}-[0-9]+$", d) is not None]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        # before we save the new checkpoint, we need to have at _most_
        # `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing "
                f"{len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)
    save_path = save_last_checkpoint_dir(args,
                                         accelerator,
                                         global_step,
                                         best_loss)
    logger.info(f"Saved state to {save_path}")

    if curr_loss is not None and (best_loss is None or curr_loss < best_loss):
        logger.info(f"best_loss: {best_loss} curr_loss: {curr_loss}")
        best_loss = curr_loss
        best_checkpoints = os.listdir(args.output_dir)
        best_checkpoints = [d for d in best_checkpoints if
                            re.search(rf"{args.checkpoint_prefix}-[0-9]+-best$", d) is not None]
        best_checkpoints = sorted(best_checkpoints, key=lambda x: int(x.split("-")[-2]))
        if len(best_checkpoints) > 0:
            logger.info(f"removing previous best checkpoint: {best_checkpoints[-1]}")
            shutil.rmtree(os.path.join(args.output_dir, best_checkpoints[-1]))
        save_path_best = save_last_checkpoint_dir(args,
                                                  accelerator,
                                                  f"{global_step}-best",
                                                  best_loss)
        logger.info(f"Saved state to {save_path_best}")
    return best_loss
    # logger.info(f"Started uploading to HuggingFace Hub")
    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
    #     executor.submit(src.utils.upload_to_hugging_face_hub.main)
    # logger.info(f"Finished uploading to HuggingFace Hub")


def save_last_checkpoint_dir(args, accelerator, global_step, best_loss):
    save_path = os.path.join(args.output_dir, f"{args.checkpoint_prefix}-{global_step}")
    accelerator.save_state(save_path)
    save_loss_path = os.path.join(args.output_dir, f"{args.checkpoint_prefix}-{global_step}")
    with open(f"{save_loss_path}/loss.json", "w") as f:
        json.dump({"best_loss": best_loss}, f, indent=4, sort_keys=True)
    args_sorted_for_print = dict(OrderedDict(sorted(vars(args).items())))
    with open(f"{save_loss_path}/hyper_params_config.json", "w") as json_file:
        json.dump(args_sorted_for_print, json_file, indent=4, sort_keys=True)
    return save_path


def get_last_checkpoint_dir(output_dir, prefix, accelerator, get_best=False, download_from_hub=False):
    dirs = os.listdir(output_dir)
    if get_best:
        dirs_to_search = [d for d in dirs if re.search(rf"{prefix}-[0-9]+-best$", d) is not None]
    else:
        dirs_to_search = [d for d in dirs if re.search(rf"{prefix}-[0-9]+(-best)?$", d) is not None]
    dirs = sorted(dirs_to_search, key=lambda x: int(x.replace("-best", "").split("-")[-1]))
    path = dirs[-1] if len(dirs) > 0 else None
    with accelerator.main_process_first() if accelerator is not None else nullcontext():
        if path is None or download_from_hub:
            # login("")
            fs = HfFileSystem()
            api = HfApi()
            files_from_repo_folder = fs.glob(f"RanG1991/inter-rep-sd-control/{Path(output_dir).name}/{prefix}/*")
            print(Path(output_dir) / (prefix + "-0"))
            for file_from_repo_folder in files_from_repo_folder:
                api.hf_hub_download(filename=Path(file_from_repo_folder).name,
                                    subfolder=f"{Path(output_dir).name}/{prefix}",
                                    repo_id="RanG1991/inter-rep-sd-control",
                                    local_dir="",
                                    force_download=True)
            # move the contents of the downloaded folder because the whole path is duplicated during
            # the download, and also add the step number to be 0
            if len(files_from_repo_folder) > 0:
                os.chmod(f"{Path(output_dir).name}/{prefix}", 0o777)
                shutil.move(f"{Path(output_dir).name}/{prefix}",
                            f"{Path(output_dir)}/{prefix}-0")
                # list_files = (os.getcwd() / Path(output_dir).name).rglob("*.*")
                path = os.path.join(output_dir, prefix + "-0")
            else:
                path = None
        else:
            path = os.path.join(output_dir, path)
    print(f"using checkpoint: {path}")
    return path


class BLIP_captioner:
    def __init__(self, device):
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                                   torch_dtype=torch.float16)
        if device is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(device)

    def __call__(self, image):
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text


class DistributedSamplerNoDuplicate(DistributedSampler):
    """ A distributed sampler that doesn't add duplicates. Arguments are the same as DistributedSampler """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # some ranks may have less samples, that's fine
            if self.rank >= len(self.dataset) % self.num_replicas:
                self.num_samples -= 1
            self.total_size = len(self.dataset)


def conditioning_image_transforms_original_controlnet(input, resolution=512):
    input = input.resize((resolution, resolution))
    transform = transforms.Compose(
        [
            # transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(resolution),
            transforms.ToTensor()
        ]
    )
    if isinstance(input, list):
        return torch.stack([transform(item) for item in input])
    else:
        return transform(input)


def convert_language_bind_depth(depth_map, modality_transform_depth, resolution=512):
    depth_map_transformed = modality_transform_depth(depth_map)
    control_images_transformed = depth_map_transformed["pixel_values"]
    control_images_transformed = torch.nn.functional.interpolate(control_images_transformed,
                                                                 size=resolution,
                                                                 mode="bicubic")
    return control_images_transformed


def image_transforms_original_controlnet(input, resolution=512):
    input = input.resize((resolution, resolution))
    transform = transforms.Compose(
        [
            # transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return transform(input)


def normalize_tensor(t):
    t_min, t_max = t.min(), t.max()
    t_normalized = ((t - t_min) / (t_max - t_min))
    return t_normalized


def from_numpy_to_tensor(array, normalize=False, resolution=512):
    t = torch.tensor(array, dtype=torch.float32).permute((2, 0, 1))
    if normalize:
        t = normalize_tensor(t)
    t = torch.nn.functional.interpolate(t, size=(resolution, resolution), mode='bicubic', align_corners=False)
    return t


def from_PIL_to_tensor(image, normalize=False, resolution=512):
    t = torch.tensor(np.asarray(image.resize((resolution, resolution))),
                     dtype=torch.float32).permute((2, 0, 1))
    if normalize:
        t = normalize_tensor(t)
    return t


def compute_inception_features(inception_model, images, dims):
    pred_arr = np.empty((len(images), dims))
    start_idx = 0
    with torch.no_grad():
        pred = inception_model(images)[0]
    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    pred = pred.squeeze(3).squeeze(2).cpu().numpy()
    pred_arr[start_idx: start_idx + pred.shape[0]] = pred
    return pred_arr


def compute_FID_metrics(real_images, fake_images, resolution, device="cuda", dims=2048):
    real_images = [transforms.ToTensor()(real_image.resize((resolution, resolution)).convert("RGB")) for real_image in
                   real_images]
    fake_images = [transforms.ToTensor()(fake_image.resize((resolution, resolution)).convert("RGB")) for fake_image in
                   fake_images]
    # save_image(real_images + fake_images, "/sci/labs/sagieb/ranga/InterSem/test_imgs/images.png")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()
    act_real = compute_inception_features(inception_model=inception_model,
                                          images=torch.stack(real_images).to(device),
                                          dims=dims)
    mu_real = np.mean(act_real, axis=0)
    sigma_real = np.cov(act_real, rowvar=False)
    act_fake = compute_inception_features(inception_model=inception_model,
                                          images=torch.stack(fake_images).to(device),
                                          dims=dims)
    mu_fake = np.mean(act_fake, axis=0)
    sigma_fake = np.cov(act_fake, rowvar=False)
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_value


def plot_images_and_prompt(images, titles, prompt):
    plt.clf()
    plt.figure(figsize=(30, 60))
    num_cols = 3
    num_rows = math.ceil((len(images) + 1) / num_cols)
    for idx, image in enumerate(images):
        plt.subplot(num_rows, num_cols, idx + 1)
        plt.title(titles[idx], fontsize=15)
        plt.axis('off')
        plt.imshow(image)
    plt.subplot(num_rows, num_cols, len(images) + 1)
    plt.text(0.5, 0.5, prompt, horizontalalignment='center', verticalalignment='center', wrap=True, fontsize=50)
    plt.axis('off')
    prompt_for_file_name = prompt.replace(' ', '_').replace('\n', '_').replace('.', '')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.tight_layout()
    plt.savefig(f"{Path(__file__).resolve().parent.parent.parent}/test_imgs/{prompt_for_file_name}.png")
    plt.close()


def split_to_train_test_validation(dataset_raw):
    train_testvalid = dataset_raw.train_test_split(test_size=0.2, shuffle=False)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, shuffle=False)
    dataset_with_splits = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']})
    return dataset_with_splits


def plot_image_and_hint(dataset_item):
    image, hint, prompt = dataset_item["jpg"], dataset_item["hint"], dataset_item["txt"]
    cv2.imwrite(f"{top_dir}/InterSem/test_imgs/img_{prompt.replace(' ', '_').replace('.', '')}.png",
                cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    imageio.imwrite(f"{top_dir}/InterSem/test_imgs/hint_{prompt.replace(' ', '_').replace('.', '')}.png",
                    np.asarray(hint))


def check_first_10_hints_in_dataset(dataset_obj):
    for i in range(10):
        plot_image_and_hint(dataset_obj[i])


def plot_generated_and_original_embeddings(generated_embeddings, orig_embeddings, titles):
    assert len(generated_embeddings) == len(orig_embeddings) == len(titles)
    for generated_embedding, orig_embedding, title in zip(generated_embeddings, orig_embeddings, titles):
        generated_embedding_to_plot = return_embedding_as_image(generated_embedding)
        orig_embedding_to_plot = return_embedding_as_image(orig_embedding)
        plot_images_and_prompt(images=[generated_embedding_to_plot, orig_embedding_to_plot],
                               titles=["generated", "original"],
                               prompt=title)


def return_embedding_as_image(embedding, index_to_plot=0, vmin=0, vmax=255):
    embedding = embedding.clone().detach()
    assert len(embedding.shape) == 3, "the shape of the embedding has to be 3 (channels, width, height)"
    embedding_to_plot = normalize_tensor(embedding) * (vmax - vmin) + vmin
    if embedding_to_plot.shape[0] > 3:
        embedding_to_plot = embedding_to_plot[index_to_plot, :, :]
    else:
        embedding_to_plot = embedding_to_plot.permute((1, 2, 0))
    embedding_to_plot = Image.fromarray(embedding_to_plot.cpu().numpy().astype(np.uint8)).convert("RGB")
    return embedding_to_plot


def create_requirements_based_on_freeze():
    with open(f"{top_dir}/InterSem/requirements.txt") as req_orig:
        with open(f"{top_dir}/InterSem/requirements_freeze.txt") as req_freeze:
            dict_req_freeze = {}
            for row in req_freeze:
                line = row.strip()
                line_split = line.split("==")
                package_name = line_split[0]
                version = line_split[1]
                dict_req_freeze[package_name] = version
            dict_req_orig = {}
            for row in req_orig:
                line = row.strip()
                line_split = line.split("==")
                package_name = line_split[0]
                dict_req_orig[package_name] = dict_req_freeze[package_name]
    with open(f"{top_dir}/InterSem/requirements_new.txt", "w") as req_new:
        for package in dict_req_orig:
            req_new.write(package + "==" + dict_req_orig[package] + "\n")


if __name__ == "__main__":
    # create_requirements_based_on_freeze()
    generate_collage_from_good_or_bad_images(r"C:\Users\galun\Desktop\images_040924\separate_controls_and_SD_for_plot")
