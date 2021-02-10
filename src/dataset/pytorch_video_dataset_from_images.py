import random
from typing import (
    Dict,
    Optional,
    Tuple
)

import cv2
import numpy as np
import torch

from .pytorch_video_dataset_utils import n_to_n_loader_from_images


class PytorchVideoDatasetFromImages(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(self, data_path: str, label_map: Dict[int, str], n_to_n: bool, sequence_length: int,
                 grayscale: bool, image_sizes: Tuple[int, int],
                 transform: Optional[type] = None, limit: Optional[int] = None, defects: Optional[list[str]] = None,
                 load_data: bool = True):
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
            defects: Filters given defects (for exemple: ["g1000", "s1000"])
            load_data: If True then all the videos are loaded into ram
        """
        self.transform = transform
        self.load_data = load_data

        self.n_to_n = n_to_n
        self.sequence_length = sequence_length
        self.grayscale = grayscale
        self.image_sizes = image_sizes

        self.data = n_to_n_loader_from_images(data_path, label_map,
                                              limit=limit, defects=defects, load_videos=load_data, grayscale=grayscale)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        start = random.randint(0, len(self.data[i, 0]) - self.sequence_length)
        if self.load_data:
            video = self.data[i, 0].astype(np.uint8)
            video = video[start:start+self.sequence_length]
        else:
            video = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                     for image_path in self.data[i, 0, start:start+self.sequence_length]]

        label = self.data[i, 1][start:start+self.sequence_length].astype(np.uint8)
        if not self.n_to_n:
            label = label = np.amax(label)

        sample = {"data": np.asarray(video, dtype=np.uint8), "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample
