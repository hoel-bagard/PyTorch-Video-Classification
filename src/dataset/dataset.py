import random

import torch
import cv2
import numpy as np

from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.dataset.dataset_utils import (
    default_loader,
    n_to_n_loader
)


class Dataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(self, data_path: str, transform=None, limit: int = None, load_videos: bool = True):
        """
        Args:
            data_path:
                Path to the root folder of the dataset.
                This folder is expected to contain subfolders for each class, with the videos inside.
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): If given then the number of elements for each class in the dataset
                                   will be capped to this number
            load_videos: If True then all the videos are loaded into ram

        """
        self.transform = transform
        self.load_videos = load_videos

        if ModelConfig.USE_N_TO_N:
            self.labels = n_to_n_loader(data_path, DataConfig.LABEL_MAP, limit=limit, load_videos=self.load_videos)
        else:
            self.labels = default_loader(data_path, DataConfig.LABEL_MAP, limit=limit, load_videos=self.load_videos)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        if self.load_videos:
            video = self.labels[i, 0].astype(np.uint8)
            start = random.randint(0, len(video) - ModelConfig.VIDEO_SIZE)
            video = video[start:start+ModelConfig.VIDEO_SIZE]
        else:
            cap = cv2.VideoCapture(self.labels[i, 0])
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            start = random.randint(0, frame_count-1 - ModelConfig.VIDEO_SIZE)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            video = []
            for j in range(ModelConfig.VIDEO_SIZE):
                frame_ok, frame = cap.read()
                if frame_ok:
                    if ModelConfig.USE_GRAY_SCALE:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = np.expand_dims(frame, -1)  # To keep a channel dimension (gray scale)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:  # frame is None for some reason
                    if ModelConfig.USE_GRAY_SCALE:
                        frame = np.zeros((*ModelConfig.IMAGE_SIZES, 1), np.uint8)
                    else:
                        frame = np.zeros((ModelConfig.IMAGE_SIZES[0], ModelConfig.IMAGE_SIZES[1], 3), np.uint8)
                video.append(frame)

            cap.release()
            video = np.asarray(video)

        if ModelConfig.USE_N_TO_N:
            label = self.labels[i, 1][start:start+ModelConfig.VIDEO_SIZE]
        else:
            label = int(self.labels[i, 1])
        sample = {"video": video, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample
