import os

import torch
from torchvision.transforms import Compose
from nvidia.dali.plugin import pytorch

from .pytorch_dataset import Dataset
from .dali_dataloader import DALILoader
from config.data_config import DataConfig
from config.model_config import ModelConfig
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
        if DataConfig.DALI:
            batch = batch[0]
        else:
            batch["video"] = batch["video"].to(self.device).float()
            batch["label"] = batch["label"].to(self.device).long()
        return batch

    def __len__(self):
        return len(self.dataloader)


class Dataloader(object):
    """Wrapper around the PyTorch / DALI DataLoaders"""
    def __init__(self, data_path: str, transform=None, limit: int = None, load_videos: bool = False):
        mode = "Train" if "Train" in data_path else "Validation"

        if DataConfig.DALI:
            self.dataloader: pytorch.DALIGenericIterator = DALILoader(data_path,
                                                                      DataConfig.LABEL_MAP,
                                                                      limit=limit,
                                                                      mode=mode)
        else:
            if mode == "Train":
                data_transforms = Compose([
                    transforms.RandomCrop(),
                    transforms.Resize(*ModelConfig.IMAGE_SIZES),
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
                    transforms.Resize(*ModelConfig.IMAGE_SIZES),
                    transforms.Normalize(),
                    transforms.ToTensor()
                ])

            dataset = Dataset(data_path,
                              limit=limit,
                              load_videos=load_videos,
                              transform=data_transforms)
            self.dataloader = torch.utils.data.DataLoader(dataset,
                                                          batch_size=ModelConfig.BATCH_SIZE,
                                                          shuffle=(mode == "Train"),
                                                          num_workers=ModelConfig.WORKERS,
                                                          drop_last=(ModelConfig.NETWORK == "LRCN"))

        msg = f"{mode} data loaded"
        print(msg + ' ' * (os.get_terminal_size()[0] - len(msg)))

    def __iter__(self):
        return DataIterator(self)

    def __len__(self):
        return len(self.dataloader)
