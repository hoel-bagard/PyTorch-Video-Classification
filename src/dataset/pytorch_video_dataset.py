import random
from pathlib import Path
from typing import (
    Dict,
    Optional,
    Tuple
)

import cv2
import numpy as np
import torch

from .pytorch_video_dataset_utils import (
    n_to_1_loader,
    n_to_n_loader
)


class PytorchVideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(self, data_path: Path, label_map: Dict[int, str], n_to_n: bool, sequence_length: int,
                 grayscale: bool, image_sizes: Tuple[int, int],
                 transform: Optional[type] = None, limit: Optional[int] = None, load_data: bool = True):
        """
        Args:
            data_path:
                Path to the root folder of the dataset.
                This folder is expected to contain subfolders for each class, with the videos inside.
            label_map: dictionarry mapping an int to a class
            n_to_n: Whether the labels / predictions are done for each frame or once per video.
            sequence_length: Length of the sequences fed to the network
            grayscale: If set to true, images will be converted to grayscale
            image_sizes: Dimension of the frames (width, height)
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): If given then the number of elements for each class in the dataset
                                   will be capped to this number
            load_data: If True then all the videos are loaded into ram
        """
        self.transform = transform
        self.load_data = load_data

        self.n_to_n = n_to_n
        self.sequence_length = sequence_length
        self.grayscale = grayscale
        self.image_sizes = image_sizes

        if n_to_n:
            self.labels = n_to_n_loader(data_path, label_map, limit=limit, load_videos=load_data, grayscale=grayscale)
        else:
            self.labels = n_to_1_loader(data_path, label_map, limit=limit, load_videos=load_data, grayscale=grayscale)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        if self.load_data:
            video = self.labels[i, 0].astype(np.uint8)
            start = random.randint(0, len(video) - self.sequence_length)
            video = video[start:start+self.sequence_length]
        else:
            cap = cv2.VideoCapture(str(self.labels[i, 0]))
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            start = random.randint(0, frame_count-1 - self.sequence_length)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            video = []
            for j in range(self.sequence_length):
                frame_ok, frame = cap.read()
                if frame_ok:
                    if self.grayscale:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = np.expand_dims(frame, -1)  # To keep a channel dimension (gray scale)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:  # frame is None for some reason
                    if self.grayscale:
                        frame = np.zeros((*self.image_sizes, 1), np.uint8)
                    else:
                        frame = np.zeros((self.image_sizes[0], self.image_sizes[1], 3), np.uint8)
                video.append(frame)

            cap.release()
            video = np.asarray(video)

        if self.n_to_n:
            label = self.labels[i, 1][start:start+self.sequence_length]
        else:
            label = int(self.labels[i, 1])
        sample = {"data": video, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample
