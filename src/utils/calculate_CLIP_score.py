import sys
import os
import pathlib

sys.path.append(os.getcwd())
top_dir = pathlib.Path(__file__).parent.parent.parent.parent

from torchmetrics.multimodal.clip_score import CLIPScore
from pathlib import Path
from PIL import Image
from src.utils.project_utils import from_PIL_to_tensor
import torch
import functools
from matplotlib import pyplot as plt
import re

sys.path.append(os.getcwd())
top_dir = pathlib.Path(__file__).parent.parent.parent.parent

FOLDERS_TO_EXCLUDE = [
    "depth_original_control_gen",
    # "seg_original_control_gen",
    "depth_hed_text_control_gen",
    "depth_hed_image_and_control",
    "depth_seg_text_control_gen",
    "depth_seg_image_and_control",
    "hed_text_control_gen",
    "hed_image_and_control",
    "seg_text_control_gen",
    "seg_image_and_control",
    "original_SD",
    "comp_for_paper",
    "depth_text_control_gen",
    "depth_image_and_control"
]


class CalculateCLIPScore:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(self.device)
        self.test_imgs_dir = Path(top_dir / "InterSem" / "test_imgs")
        self.folder_names_list = [self.test_imgs_dir / dir for dir in os.listdir(path=self.test_imgs_dir) if
                                  os.path.isdir(
                                      os.path.join(self.test_imgs_dir, dir)) and dir not in FOLDERS_TO_EXCLUDE]

    def calculate_CLIP_score_between_images_and_texts(self, images, texts):
        with torch.no_grad():
            score = self.metric(images, texts)
            # fig_, ax_ = self.metric.plot(values)
            # plt.close()
        return score.detach().item()

    def get_common_files_in_folders(self, folders_list_test_imgs):
        files_all_folders = []
        for folder in folders_list_test_imgs:
            if folder.name not in FOLDERS_TO_EXCLUDE:
                all_files = [file.name for file in folder.glob("*.*")]
                files_all_folders.append(set(all_files))
        files_to_include = list(functools.reduce(lambda x, y: x.intersection(y), files_all_folders))
        print(f"there are {len(files_to_include)} common files in all folders")
        return files_to_include

    def calculate_CLIP_score_for_folder(self, folder_path, files_to_include):
        all_image_files = list(Path(folder_path).glob("*.*"))
        scores = []
        for i in range((len(all_image_files) // 100) + 1):
            image_list = all_image_files[i * 100: (i + 1) * 100]
            text_list = []
            image_as_tensor_list = []
            for image_file in image_list:
                if image_file.name in files_to_include:
                    text = " ".join(image_file.stem.split("_"))
                    image_tensor = from_PIL_to_tensor(Image.open(image_file)).to("cuda")
                    text_list.append(text)
                    image_as_tensor_list.append(image_tensor)
            if len(image_as_tensor_list) > 0:
                score = self.calculate_CLIP_score_between_images_and_texts(image_as_tensor_list, text_list)
                scores.append(score)
                print(f"This is {i + 1} of {(len(all_image_files) // 100) + 1}", flush=True)
        print(f"the number of scores in {folder_path} is: {len(scores)}", flush=True)
        return sum(scores) / len(scores)

    def calculate_CLIP_score(self):
        dict_folder_name_to_score = {}
        file_to_include = self.get_common_files_in_folders(self.folder_names_list)
        for folder_path in self.folder_names_list:
            score = self.calculate_CLIP_score_for_folder(folder_path, file_to_include)
            dict_folder_name_to_score[folder_path.name] = score
        dict_folder_name_to_score = sorted(dict_folder_name_to_score.items(), key=lambda x: x[1], reverse=True)
        for folder_path, score in dict_folder_name_to_score:
            print(f"The CLIP score of {folder_path} is: {score}")

    def get_images_with_better_and_worse_CLIP_score(self):
        comp_for_paper_folder = Path(f"{top_dir}/InterSem/test_imgs/comp_for_paper")
        comp_for_paper_folder.mkdir(parents=True, exist_ok=True)
        file_to_include = self.get_common_files_in_folders(self.folder_names_list)
        for file_path in file_to_include:
            file_and_folder_name_to_score = {}
            file_name_wo_ext = Path(file_path).stem
            for folder_path in self.folder_names_list:
                image = Image.open(folder_path / file_path)
                folder_path_image_and_control = Path(str(folder_path).replace("text_control_gen",
                                                                              "image_and_control"))
                if not folder_path_image_and_control.exists():
                    raise Exepction(f"The folder with path: {folder_path_image_and_control} does not exist")
                image_and_control = Image.open(folder_path_image_and_control / file_path)
                text = " ".join(file_name_wo_ext.split("_"))
                image_tensor = from_PIL_to_tensor(image).to("cuda")
                score = self.calculate_CLIP_score_between_images_and_texts([image_tensor], [text])
                file_and_folder_name_to_score[folder_path.name + "_" + file_name_wo_ext] = (
                    score, image, image_and_control)
            (score_seg, image_file_seg, image_and_control_seg) = file_and_folder_name_to_score[
                "seg_text_control_gen_" + file_name_wo_ext]
            (score_depth, image_file_depth, image_and_control_depth) = file_and_folder_name_to_score[
                "depth_text_control_gen_" + file_name_wo_ext]
            (score_hed, image_file_hed, image_and_control_hed) = file_and_folder_name_to_score[
                "hed_text_control_gen_" + file_name_wo_ext]
            (score_SD, image_file_SD, _) = file_and_folder_name_to_score[
                "original_SD_" + file_name_wo_ext]
            if score_SD <= score_seg - 3 and score_SD <= score_depth - 3 and score_SD <= score_hed - 3:
                print(f"The CLIP score of seg of file {file_name_wo_ext} is {score_seg}\n"
                      f"The CLIP score of depth of file {file_name_wo_ext} is {score_depth}\n"
                      f"The CLIP score of hed of file {file_name_wo_ext} is {score_hed}\n"
                      f"The CLIP score of SD of file {file_name_wo_ext} is {score_SD}")
                image_and_control_seg.resize((1024, 512)).save(
                    (comp_for_paper_folder / Path("seg_text_control_gen_" + str(file_path))))
                image_and_control_depth.resize((1024, 512)).save(
                    (comp_for_paper_folder / Path("depth_text_control_gen_" + str(file_path))))
                image_and_control_hed.resize((1024, 512)).save(
                    (comp_for_paper_folder / Path("hed_text_control_gen_" + str(file_path))))
                image_file_SD.resize((512, 512)).save(
                    (comp_for_paper_folder / Path("SD_text_control_gen_" + str(file_path))))


if __name__ == "__main__":
    clip_score_calc = CalculateCLIPScore()
    # clip_score_calc.get_images_with_better_and_worse_CLIP_score()
    clip_score_calc.calculate_CLIP_score()
