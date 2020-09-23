import os
import glob
import random

import torch
import cv2
import numpy as np
from PIL import Image

from config.model_config import ModelConfig


class Dataset(torch.utils.data.Dataset):
    """Classification dataset."""

    def __init__(self, data_path: str, transform=None):
        """
        Args:
            data_path:
                Path to the root folder of the dataset.
                This folder is expected to contain subfolders for each class, with the videos inside.
                It should also contain a "class.names" with all the classes
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.video_size = ModelConfig.VIDEO_SIZE

        self.label_map = DataConfig.LABEL_MAP

        labels = []
        for key in range(len(self.label_map)):
            for video_path in glob.glob(os.path.join(data_path, self.label_map[key], "*.avi")):
                print(f"Loading data {video_path}   ", end="\r")
                labels.append([video_path, key])        

        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        label = int(self.labels[i, 1])
        cap = cv2.VideoCapture(self.labels[i, 0])

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        start = random.randint(0, frame_count-2 - self.video_size)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        video = []
        for j in range(self.video_size):
            frame_ok, frame = cap.read()
            if frame_ok:
                if ModelConfig.USE_GRAY_SCALE:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = np.expand_dims(frame, -1)  # To keep a channel dimension (gray scale)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:  # frame is None for some reason
                # print(f"\nFrame was none: {frame}, video: {self.labels[i, 0]}")
                # print(f"Frame count was: {frame_count}, batch frame: {j}")
                # print(f"Frame: {cap.get(cv2.CAP_PROP_POS_FRAMES)}, start: {start}")
                if ModelConfig.USE_GRAY_SCALE:
                    frame = np.zeros((*ModelConfig.IMAGE_SIZES, 1), np.uint8)
                else:
                    frame = np.zeros((ModelConfig.IMAGE_SIZES[0], ModelConfig.IMAGE_SIZES[1], 3), np.uint8)
            video.append(frame)
        cap.release()

        sample = {"video": np.asarray(video), "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample
