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

        # Build a map between id and names
        self.label_map = {}
        with open(os.path.join(data_path, "..", "classes.names")) as table_file:
            for key, line in enumerate(table_file):
                label = line.strip()
                self.label_map[key] = label

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

        cap = cv2.VideoCapture(self.labels[i, 0])

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, frame_count-1 - self.video_size))

        video = []
        for _ in range(self.video_size):
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
                frame = frame.unsqueeze(0)
            video.append(frame)

        cap.release()
        video = torch.cat(video, 0)
        label = torch.from_numpy(np.asarray(self.labels[i, 1], dtype=np.uint8))
        sample = {'video': video, 'label': label}
        return sample
