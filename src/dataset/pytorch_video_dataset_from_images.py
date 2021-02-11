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
                 transform: Optional[type] = None, limit: Optional[int] = None, filters: Optional[list[str]] = None,
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
            filters: Filters data whose path include the given filters (for exemple: ["subfolder1", "class2"])
            load_data: If True then all the videos are loaded into ram
        """
        self.transform = transform
        self.load_data = load_data

        self.n_to_n = n_to_n
        self.sequence_length = sequence_length
        self.grayscale = grayscale
        self.image_sizes = image_sizes

        self.data, self.labels = n_to_n_loader_from_images(data_path, label_map, sequence_length,
                                                           limit=limit, filters=filters, load_videos=load_data,
                                                           grayscale=grayscale)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        if self.load_data:
            video = np.asarray(self.data[i], dtype=np.uint8)
        else:
            video = np.asarray([cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in self.data[i]], dtype=np.uint8)

        label = self.labels[i]
        if not self.n_to_n:
            label = label = np.amax(label)

        sample = {"data": video, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample
