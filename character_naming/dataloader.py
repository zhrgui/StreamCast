import ast
import os
import json
import cv2
import decord
import pickle
import pandas as pd
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from insightface.app import FaceAnalysis
from promptloader import get_prompt_template

import warnings
warnings.simplefilter("ignore", FutureWarning)

class Name_Linking_Dataset_Omni(Dataset):
    def __init__(self, video_dir, anno_file, prompt_id=4, audio_dir=None):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.annotations = pd.read_csv(anno_file)
        self.prompt_id = prompt_id
        self.prompt_template = get_prompt_template(prompt_id)
        self.use_audio_only = audio_dir is not None
        if self.use_audio_only:
            assert prompt_id == 7, "prompt_id must be 7 when using audio only"

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        track_color_map = ast.literal_eval(row["color_map"])
        color_list = sorted(list(set(track_color_map.values())))
        candidate_names = ast.literal_eval(row["name"])

        fmt_kwargs = {"candidate_character_names": ", ".join(candidate_names), "box_colors": ", ".join(color_list)}

        if "{timestamped_subtitles}" in self.prompt_template:
            start = row["start"]
            subtitles = ast.literal_eval(row["subtitles"])
            subtitle_lines = []
            for entry in subtitles:
                entry_start = entry["start"] - start
                entry_end = entry["end"] - start
                text = entry["text"]
                speaker = entry["speaker"] if entry["speaker"] != "none" else "unknown"
                subtitle_lines.append(f"<{entry_start:.2f} seconds> [{speaker}]: {text.strip()} <{entry_end:.2f} seconds>")
            fmt_kwargs["timestamped_subtitles"] = "\n".join(subtitle_lines)

        prompt = self.prompt_template.format(**fmt_kwargs)

        if self.use_audio_only:
            audio_path = os.path.join(self.audio_dir, f"{index}.wav")
            return audio_path, prompt
        else:
            video_path = os.path.join(self.video_dir, f"{index}.mp4")
            return video_path, prompt

class Name_Linking_Dataset_Video(Dataset):
    def __init__(self, video_dir, anno_file, prompt_id):
        self.video_dir = video_dir
        self.annotations = pd.read_csv(anno_file)
        self.prompt_template = get_prompt_template(prompt_id)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        start = row["start"]
        video_path = os.path.join(self.video_dir, row["video_path"])
        track_color_map = ast.literal_eval(row["color_map"])
        color_list = sorted(list(set(track_color_map.values())))
        candidate_names = ast.literal_eval(row["name"])
        subtitles = ast.literal_eval(row["subtitles"])

        def format_subtitles(subtitles, start):
            subtitle_lines = []
            for entry in subtitles:
                entry_start = entry["start"] - start
                entry_end = entry["end"] - start
                text = entry["text"]
                speaker = entry["speaker"] if entry["speaker"] != "none" else "unknown"
                subtitle_lines.append(f"<{entry_start:.2f} seconds> [{speaker}]: {text.strip()} <{entry_end:.2f} seconds>")
            return "\n".join(subtitle_lines)
            
        timestamped_subtitles = format_subtitles(subtitles, start)
        prompt = self.prompt_template.format(candidate_character_names=", ".join(candidate_names), box_colors=", ".join(color_list), timestamped_subtitles=timestamped_subtitles)
        return video_path, prompt
