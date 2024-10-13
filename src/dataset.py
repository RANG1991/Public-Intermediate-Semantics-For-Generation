import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

top_dir = Path(__file__).parent.parent.parent.absolute()

os.environ["HF_HOME"] = f"{top_dir}/huggingface/"
os.environ["HF_DATASETS_CACHE"] = f"{top_dir}/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{top_dir}/huggingface/models"
os.environ["TORCH_HOME"] = f"{top_dir}/torch"

from datasets import load_dataset, load_from_disk, DatasetDict
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch.multiprocessing
import seaborn as sns
import matplotlib
import math
from PIL import Image
from src.utils import project_utils
from ControlnetGithub.annotator.uniformer import UniformerDetector
from ControlnetGithub.annotator.canny import CannyDetector
from ControlnetGithub.annotator.hed import HEDdetector, nms
from ControlnetGithub.annotator.util import HWC3
import json
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import requests
import multiprocessing
from functools import partial
import imagehash


# from huggingface_hub import login
#
# login()


class InterSemDataset(Dataset):
    def __init__(self, split, dataset_size=-1, device="cuda", dataset_to_use="LAION", only_gt_image_and_prompt=False):
        self.device = device
        self.only_gt_image_and_prompt = only_gt_image_and_prompt
        self.dataset_to_use = dataset_to_use
        if dataset_to_use == "fill50k":
            dataset_raw = load_dataset("HighCWu/fill50k", split="train")
            dataset_with_splits = project_utils.split_to_train_test_validation(dataset_raw=dataset_raw)
            self.data = dataset_with_splits[split]
        elif dataset_to_use == "COCO":
            if Path("/cs/dataset/COCO/").exists():
                if split == "validation":
                    split = "val"
                self.dataset_path = f"{top_dir}/captions_{split}2017.json"
                self.data = json.load(open(self.dataset_path))
                # if split == "val":
                #     item_to_start = self.data[12]
                #     self.data[12] = self.data[0]
                #     self.data[0] = item_to_start
            else:
                if split == "val":
                    split = "validation"
                self.data = load_dataset(str(Path(top_dir / "COCO_dataset" / "coco_dataset_script.py").absolute()),
                                         "2017",
                                         data_dir=top_dir / "COCO_dataset",
                                         split=split)
        elif dataset_to_use == "LAION":
            if split == "val":
                split = "validation"
            self.dataset_path = f"{top_dir}/captions_LAION_{split}2017.json"
            if not Path(self.dataset_path).exists() or not Path(f"{top_dir}/LAION/{split}").exists():
                self.create_dataset_LAION()
            self.data = json.load(open(self.dataset_path))
        else:
            raise Exception(f"The requested dataset {dataset_to_use} is not supported")
        if dataset_size != -1:
            if dataset_size > len(self.data):
                print("The requested dataset size is greater then the dataset length!")
            dataset_size = min(dataset_size, len(self.data))
            # self.data = self.data.shuffle(seed=90)
            if self.dataset_to_use == "fill50k" or not hasattr(self, "dataset_path") or not Path(
                    self.dataset_path).exists():
                self.data = self.data.select(list(range(0, dataset_size)))
            else:
                self.data = self.data[:dataset_size]
        self.split = split
        self.model_depth = None
        # self.apply_canny = CannyDetector()
        self.apply_hed = HEDdetector()
        # self.model_sketch = cv2.dnn.readNetFromCaffe(f"{top_dir}/models/deploy.prototxt",
        #                                              f"{top_dir}/models/hed_pretrained_bsds.caffemodel")
        self.apply_uniformer = UniformerDetector()
        self.all_captions_set = set()
        # self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(
        #     "cuda")

    def __len__(self):
        return len(self.data)

    def get_sketch(self, input_image):
        img = np.asarray(input_image)
        (H, W) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H),
                                     swapRB=False, crop=False)
        self.model_sketch.setInput(blob)
        hed = self.model_sketch.forward()
        hed = cv2.resize(hed[0, 0], (W, H))
        hed = (255 * hed).astype("uint8")
        sketch_image = Image.fromarray(hed).convert("RGB")
        return sketch_image

    def get_hed(self, input_image):
        with torch.no_grad():
            input_image = np.asarray(input_image)
            hed_img = self.apply_hed(input_image)
            hed_img = Image.fromarray(HWC3(hed_img))
        return hed_img

    def get_scribble(self, input_image):
        input_image = np.asarray(input_image)
        scribble_img = np.zeros_like(input_image, dtype=np.uint8)
        scribble_img[np.min(input_image, axis=2) < 127] = 255
        scribble_img = Image.fromarray(HWC3(scribble_img))
        return scribble_img

    def get_scribble_hed(self, input_image):
        with torch.no_grad():
            input_image = np.asarray(input_image)
            input_image = HWC3(input_image)
            detected_map = self.apply_hed(input_image)
            detected_map = HWC3(detected_map)
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0
            scribble_img = Image.fromarray(detected_map)
        return scribble_img

    def get_contours(self, input_image):
        input_image = np.asarray(input_image)
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Remove dotted lines
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 5000:
                cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
        # Fill contours
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        close = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=6)
        cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 15000:
                cv2.drawContours(close, [c], -1, (0, 0, 0), -1)
        # Smooth contours
        close = 255 - close
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, open_kernel, iterations=3)
        contours = Image.fromarray(opening).convert("RGB")
        return contours

    def get_seg(self, input_image):
        with torch.no_grad():
            input_image = np.asarray(input_image)
            input_image = HWC3(input_image)
            seg_img = self.apply_uniformer(input_image)
            seg_img = Image.fromarray(seg_img)
        return seg_img

    def get_canny(self, input_image):
        input_image = np.asarray(input_image)
        canny_img = self.apply_canny(input_image, 10, 200)
        canny_img = Image.fromarray(HWC3(canny_img))
        return canny_img

    def load_model_depth(self):
        if self.model_depth is None:
            encoder = 'vits'  # can also be 'vitb' or 'vitl'
            self.model_depth = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).to(
                self.device).eval()
            self.depth_transform = Compose([
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
            self.model_depth.eval()
            self.model_depth.to(self.device)

    def convert_image_to_depth_map(self, input_image, prompt, save_image=False):
        # input_image = Image.open(f"../LanguageBind/assets/image/0.jpg")
        # depth_map_expected_result = np.asarray(Image.open(f"../LanguageBind/assets/depth/0.png"))
        self.load_model_depth()
        open_cv_image = np.array(input_image)
        image = open_cv_image / 255.0
        h, w = image.shape[:2]
        image = self.depth_transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            depth = self.model_depth(image)
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / depth.max()
        depth = depth.cpu().numpy()
        output = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        depth_numpy_language_bind = (output * 1000).astype("uint32")
        depth_as_image = Image.fromarray((output * 255).clip(0, 255).astype("uint8")).convert("RGB")
        if save_image:
            cv2.imwrite(f"../heatmap_and_depth_map_images/{prompt.replace(' ', '_').replace('.', '')}_depth_map.png",
                        depth_as_image)
        return depth_as_image, depth_numpy_language_bind

    @staticmethod
    def convert_image_to_heatmap(input_image, prompt, save_image=False):
        colormap = plt.get_cmap('inferno')
        heatmap = (colormap(input_image) * 2 ** 16).astype(np.uint16)[:, :, :3]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        if save_image:
            cv2.imwrite(f"../heatmap_and_depth_map_images/{prompt.replace(' ', '_').replace('.', '')}_heatmap.png",
                        heatmap)
        return heatmap

    def add_depth_map_to_item(self, example):
        example["hint"] = self.convert_image_to_depth_map(example['image'].convert('RGB'), "", save_image=False)
        return example

    def __getitem__(self, idx):
        if self.dataset_to_use == "fill50k":
            item = self.data[idx]
            image = item['image']
            image = image.convert("RGB")
            return dict(jpg=image,
                        text=item['text'],
                        depth=item['guide'].convert('L'),
                        canny=self.get_canny(image))
        elif self.dataset_to_use == "COCO":
            if hasattr(self, 'dataset_path') and Path(self.dataset_path).exists():
                prompt = self.data[idx]["caption"]
                image_id = str(self.data[idx]["image_id"]).rjust(12, '0')
                if self.split == "test" or self.split == "val":
                    image = Image.open(f"/cs/dataset/COCO/val2017/{image_id}.jpg")
                else:
                    image = Image.open(f"/cs/dataset/COCO/train2017/{image_id}.jpg")
            else:
                prompt = self.data[idx]["caption"]
                # if idx == 0 and self.split == "test":
                #     prompt = "A child holding a flowered umbrella and petting a yak."
                image = self.data[idx]["image_path"]
        elif self.dataset_to_use == "LAION":
            prompt = self.data[idx]["caption"]
            image_id = str(self.data[idx]["id"])
            if self.split == "test" or self.split == "val":
                image = Image.open(f"{top_dir}/LAION/{self.split}/{image_id}.jpg")
            else:
                image = Image.open(f"{top_dir}/LAION/{self.split}/{image_id}.jpg")
        image = image.convert("RGB")
        if prompt in self.all_captions_set:
            prompt = prompt + "."
        self.all_captions_set.add(prompt)
        depth_as_image, depth_lb = self.convert_image_to_depth_map(image, prompt, False)
        if self.only_gt_image_and_prompt:
            return dict(jpg=image,
                        text=prompt,
                        depth=None,
                        depth_lb=None,
                        seg=None,
                        # canny=self.get_canny(image),
                        hed=None)
        else:
            return dict(jpg=image,
                        text=prompt,
                        depth=depth_as_image,
                        depth_lb=depth_lb,
                        seg=self.get_seg(image),
                        # canny=self.get_canny(image),
                        hed=self.get_hed(image))

    def get_imagenet_dataset(self, root_for_jsons, root_for_data_dir):
        dict_samples_and_targets = {"validation": [], "train": []}
        syn_to_class = {}
        with open(os.path.join(root_for_jsons, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                syn_to_class[v[0]] = v[1]
        with open(os.path.join(root_for_jsons, f"ILSVRC2012_val_labels.json"), "rb") as f:
            val_to_syn = json.load(f)
            val_to_syn = {v + "_" + k.replace("ILSVRC2012_val_", "").lstrip("0"): v for k, v in val_to_syn.items()}
        samples_dir = os.path.join(root_for_data_dir)
        list_dir = os.listdir(samples_dir)
        for ind, entry in enumerate(list_dir):
            print(f"reading {ind + 1} out of {len(list_dir)}")
            syn_id = entry
            if syn_id in syn_to_class.keys():
                target = syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    dict_sample_and_target = {}
                    sample_path = os.path.join(syn_folder, sample)
                    dict_sample_and_target["image_id"] = sample_path
                    dict_sample_and_target["caption"] = target.replace("_", " ")
                    if sample in val_to_syn.keys():
                        dict_samples_and_targets["validation"].append(dict_sample_and_target)
                    else:
                        dict_samples_and_targets["train"].append(dict_sample_and_target)
        with open(os.path.join(root_for_jsons, "imagenet_id_to_label.json"), "w") as f:
            json.dump(dict_samples_and_targets, f)
        return dict_samples_and_targets

    @staticmethod
    def create_dataset_COCO():
        data_train = json.load(open(f"/cs/dataset/COCO/captions_train2017.json"))["annotations"]
        data_val = json.load(open(f"/cs/dataset/COCO/captions_val2017.json"))["annotations"]
        with open(f"{top_dir}/captions_val2017.json", "w") as f:
            json.dump(data_val[:len(data_val) // 2], f)
        with open(f"{top_dir}/captions_test2017.json", "w") as f:
            json.dump(data_val[len(data_val) // 2:], f)
        with open(f"{top_dir}/captions_train2017.json", "w") as f:
            json.dump(data_train, f)

    @staticmethod
    def create_dataset_LAION():
        dataset_raw = load_dataset("visheratin/laion-coco-nllb", split="train")
        dataset_with_splits = project_utils.split_to_train_test_validation(dataset_raw=dataset_raw)
        for split in ["train", "validation", "test"]:
            os.makedirs(f"{top_dir}/LAION/{split}", exist_ok=True)
            list_captions = []
            data = dataset_with_splits[split]
            list_indices = list(range(0, len(data)))
            if not Path(f"{top_dir}/captions_LAION_{split}2017.json").exists():
                for idx in list_indices:
                    item = data[idx]
                    list_captions.append({"caption": item["eng_caption"], "id": item['id']})
                with open(f"{top_dir}/captions_LAION_{split}2017.json", "w") as f:
                    json.dump(list_captions, f)
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                func = partial(InterSemDataset.save_laion_image, data, split)
                p.map(func, list_indices)

    @staticmethod
    def save_laion_image(data, split, idx):
        item = data[idx]
        if not Path(f"{top_dir}/LAION/{split}/{item['id']}.jpg").exists():
            image = Image.open(requests.get(f"https://nllb-data.com/{item['id']}.jpg", stream=True).raw)
            image.convert('RGB').save(f"{top_dir}/LAION/{split}/{item['id']}.jpg")
            image.close()

def ImgLabels(N, ax, images, prompts):
    def offset_image(coord, ax):
        nonlocal images
        image = images[i]
        im = matplotlib.offsetbox.OffsetImage(image, zoom=0.2)
        im.image.axes = ax
        for co, xyb in [((coord, 0), (+80, +80))]:
            ab = matplotlib.offsetbox.AnnotationBbox(im, co, xybox=xyb,
                                                     frameon=False, xycoords='data', boxcoords="offset points")
            ax.add_artist(ab)

    def offset_prompt(coord, ax):
        nonlocal prompts
        prompt = prompts[i]
        offsetbox = matplotlib.offsetbox.TextArea(project_utils.split_prompt(prompt),
                                                  textprops={"fontsize": 15, "ha": "center"})
        for co, xyb in [((0, coord), (-100, -76))]:
            ab = matplotlib.offsetbox.AnnotationBbox(offsetbox, co, xybox=xyb,
                                                     frameon=False, xycoords='data', boxcoords="offset points",
                                                     annotation_clip=False)
            ax.add_artist(ab)

    for i, c in enumerate(range(len(images))):
        offset_image(i, ax)
    for i, c in enumerate(range(len(prompts))):
        offset_prompt(i, ax)


def HeatMap(similarity_matrix):
    sns.set_theme(style="white")
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(similarity_matrix, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, annot=True, xticklabels=False, yticklabels=False)


def plot_similarity_matrices_in_grid(similarity_images, original_image, prompts):
    prompts = ["original image"] + prompts
    similarity_images = [original_image] + similarity_images
    num_cols = 2
    num_rows = math.ceil(len(similarity_images) / num_cols)
    fig, axarr = plt.subplots(num_rows, num_cols, figsize=(50, 50))
    axarr_list = list(axarr.flat)
    for i in range(len(prompts), len(axarr_list)):
        fig.delaxes(axarr_list[i])
        del axarr_list[i]
    for i, ax in enumerate(axarr_list):
        if i > 0:
            im = ax.imshow(similarity_images[i], cmap='inferno')
            ax.axis('off')
        else:
            im = ax.imshow(similarity_images[i])
            ax.axis('off')
        ax.set_title(prompts[i].replace("</w>", ""), fontsize=50)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    ticklabs = cbar_ax.get_yticklabels()
    cbar_ax.set_yticklabels(ticklabs, fontsize=50)
    fig.colorbar(im, cax=cbar_ax)
    # plt.tight_layout()
    fig.savefig(f"../test_imgs/similarity_matrices.png")


def check_language_bind_on_images_depths_prompts():
    dataset = InterSemDataset("validation", dataset_to_use="COCO", dataset_size=-1)
    dataset.data = (dataset.data.select(np.arange(start=0, stop=len(dataset), step=5)).shuffle()
                    .select([0, 1, 2, 3, 4]))
    dataset.data.cleanup_cache_files()
    images = []
    hints = []
    prompts = []
    for item in dataset:
        prompts.append(item["text"])
        images.append(item["jpg"])
        hints.append(item["depth"])
    run_language_bind_on_images_depths_prompts(images, hints, prompts)


def add_depth_maps_to_dataset():
    dict_split_to_dataset_size = {"train": 150000, "validation": 1000, "test": 1000}
    for split in dict_split_to_dataset_size:
        dataset = InterSemDataset(split, dataset_to_use="COCO", dataset_size=dict_split_to_dataset_size[split])
        dataset_with_depth_maps = dataset.data.map(lambda example: dataset.add_depth_map_to_item(example))
        dataset_with_depth_maps.save_to_disk(f"{top_dir}/COCO_with_control_net_depth_{split}")


def check_if_splits_have_different_images():
    dataset_train = InterSemDataset(split="train", dataset_to_use="COCO", dataset_size=-1)
    dataset_val = InterSemDataset(split="validation", dataset_to_use="COCO", dataset_size=-1)
    # dataset_test = InterSemDataset(split="test", dataset_to_use="COCO", dataset_size=-1)
    dataset_train_hashes_dict = {}
    dataset_val_hashes_dict = {}
    dataset_test_hashes_dict = {}
    for i, dataset_item in enumerate(dataset_train):
        dataset_train_hashes_dict[imagehash.average_hash(dataset_item["jpg"])] = (i, dataset_item["text"])
    for i, dataset_item in enumerate(dataset_val):
        dataset_val_hashes_dict[imagehash.average_hash(dataset_item["jpg"])] = (i, dataset_item["text"])
    # for i, dataset_item in enumerate(dataset_test):
    #     dataset_test_hashes_dict[imagehash.average_hash(dataset_item["jpg"])] = (i, dataset_item["txt"])
    dataset_train_hashes_set = set(dataset_train_hashes_dict.keys())
    dataset_val_hashes_set = set(dataset_val_hashes_dict.keys())
    dataset_test_hashes_set = set(dataset_test_hashes_dict.keys())
    inter_train_val = dataset_train_hashes_set.intersection(dataset_val_hashes_set)
    inter_train_test = dataset_train_hashes_set.intersection(dataset_test_hashes_set)
    inter_val_test = dataset_val_hashes_set.intersection(dataset_test_hashes_set)
    print(inter_train_val)
    print(inter_train_test)
    print(inter_val_test)
    for item in inter_train_val:
        print("train_val: {}\t{}".format(dataset_train_hashes_dict[item][1], dataset_val_hashes_dict[item][1]))
    for item in inter_train_test:
        print("train_test: {}\t{}".format(dataset_train_hashes_dict[item][1], dataset_test_hashes_dict[item][1]))
    for item in inter_val_test:
        print("val_test: {}\t{}".format(dataset_val_hashes_dict[item][1], dataset_test_hashes_dict[item][1]))


if __name__ == "__main__":
    # check_prompt_and_corresponding_image()
    # check_language_bind_on_images_depths_prompts()
    # InterSemDataset.create_dataset_LAION()
    # add_depth_maps_to_dataset()
    # project_utils.check_if_splits_have_different_images(InterSemDataset)
    check_if_splits_have_different_images()
