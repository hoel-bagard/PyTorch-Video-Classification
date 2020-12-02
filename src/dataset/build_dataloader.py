import os
from typing import (
    Dict,
    Tuple,
    Optional
)

import torch
from torchvision.transforms import Compose
from nvidia.dali.plugin import pytorch

from .pytorch_dataset import VideoDataset
from .dali_dataloader import DALILoader
import src.dataset.pytorch_transforms as transforms


class DataIterator(object):
    """An iterator."""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader.dataloader)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # TODO: remove the hardcoded 0 ?

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.dataloader_iter.next()
        # DALI has an extra dimension (don't know why), and the data is already on GPU.
        if self.dataloader.__class__.__name__ == "DALILoader":
            batch = batch[0]
        else:
            batch["video"] = batch["video"].to(self.device).float()
            batch["label"] = batch["label"].to(self.device).long()
        return batch

    def __len__(self):
        return len(self.dataloader)


class Dataloader(object):
    """Wrapper around the PyTorch / DALI DataLoaders"""
    def __init__(self, data_path: str, dali: bool, label_map: Dict[int, str], image_sizes: Tuple[int, int],
                 batch_size: int, num_workers: int, drop_last: bool = False,
                 limit: Optional[int] = None, load_data: bool = False, **model_config):
        """
        Args:
            data_path: Path to the data to load
            dali: True to use a dali dataloader, otherwise a PyTorch DataLoader will be used
            label_map: dictionarry mapping an int to a class
            image_sizes: Size of the input images
            batch_size: Batch size to use
            num_workers: Number of workers for the PyTorch DataLoader
            drop_last: Wether to drop last elements to get "perfect" batches, should be True for LSTMs
            limit: If not None then at most that number of elements will be used
            load_videos: If true then the videos will be loaded in RAM (when dali is set to False)
        """
        mode = "Train" if "Train" in data_path else "Validation"

        if dali:
            self.dataloader: pytorch.DALIGenericIterator = DALILoader(data_path,
                                                                      label_map,
                                                                      limit=limit,
                                                                      mode=mode)
        else:
            if mode == "Train":
                data_transforms = Compose([
                    transforms.RandomCrop(),
                    transforms.Resize(*image_sizes),
                    transforms.Normalize(),
                    transforms.VerticalFlip(),
                    transforms.HorizontalFlip(),
                    transforms.Rotate180(),
                    transforms.ReverseTime(),
                    transforms.ToTensor(),
                    transforms.Noise()
                ])
            else:
                data_transforms = Compose([
                    transforms.Resize(*image_sizes),
                    transforms.Normalize(),
                    transforms.ToTensor()
                ])

            dataset = VideoDataset(data_path, label_map, model_config["n_to_n"], model_config["sequence_length"],
                                   model_config["grayscale"], image_sizes, transform=data_transforms, limit=limit,
                                   load_data=load_data)
            self.dataloader = torch.utils.data.DataLoader(dataset,
                                                          batch_size=batch_size,
                                                          shuffle=(mode == "Train"),
                                                          num_workers=num_workers,
                                                          drop_last=drop_last)

        msg = f"{mode} data loaded"
        print(msg + ' ' * (os.get_terminal_size()[0] - len(msg)))

    def __iter__(self):
        return DataIterator(self)

    def __len__(self):
        return len(self.dataloader)
