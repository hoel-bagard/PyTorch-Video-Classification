import shutil
from pathlib import Path
from typing import (
    Dict,
    Tuple,
    Optional
)

import torch
from torchvision.transforms import Compose
# from nvidia.dali.plugin import pytorch

from .pytorch_video_dataset import PytorchVideoDataset
from .pytorch_video_dataset_from_images import PytorchVideoDatasetFromImages as DatasetFromImages
# from .dali_video_dataloader import DALIVideoLoader
from .pytorch_video_transforms import VideoTransforms


class VideoDataloader(object):
    """Wrapper around the PyTorch / DALI video dataLoaders"""
    def __init__(self, data_path: Path, dali: bool, load_from_images: bool, label_map: Dict[int, str],
                 image_sizes: Tuple[int, int], batch_size: int, num_workers: int,
                 dali_device_id: int = 0, drop_last: bool = False, limit: Optional[int] = None,
                 defects: Optional[list[str]] = None, load_data: bool = False, **model_config):
        """
        Args:
            data_path: Path to the data to load
            dali: True to use a dali dataloader, otherwise a PyTorch DataLoader will be used
            load_from_images: mutually exclusive with dali. Use if the dataset is made of sequences of images.
            label_map: dictionary mapping an int to a class
            image_sizes: Size of the input images
            batch_size: Batch size to use
            num_workers: Number of workers for the PyTorch DataLoader
            dali_device_id: If using DALI, which GPU to use
            drop_last: Wether to drop last elements to get "perfect" batches, should be True for LSTMs
            limit: If not None then at most that number of elements will be used
            defects: Filters given defects (for exemple: ["g1000", "s1000"])
            load_data: If true then the videos will be loaded in RAM (when dali is set to False)
        """
        self.dali: bool = dali
        self.n_to_n: bool = model_config["n_to_n"]
        mode = "Train" if "Train" in str(data_path) else "Validation"

        if dali:
            print("\nDALI dataloading is broken, plase use the PyTorch one")
            exit()
            # self.dataloader: pytorch.DALIGenericIterator = DALIVideoLoader(str(data_path),
            #                                                                label_map,
            #                                                                model_config["sequence_length"],
            #                                                                batch_size,
            #                                                                limit=limit,
            #                                                                mode=mode)
        else:
            if mode == "Train":
                data_transforms = Compose([
                    VideoTransforms.RandomCrop(),
                    VideoTransforms.Resize(*image_sizes),
                    VideoTransforms.Normalize(),
                    VideoTransforms.VerticalFlip(),
                    VideoTransforms.HorizontalFlip(),
                    VideoTransforms.Rotate180(),
                    VideoTransforms.ReverseTime(),
                    VideoTransforms.ToTensor(),
                    VideoTransforms.Noise()
                ])
            else:
                data_transforms = Compose([
                    VideoTransforms.Resize(*image_sizes),
                    VideoTransforms.Normalize(),
                    VideoTransforms.ToTensor()
                ])

            if load_from_images:
                self.dataset = DatasetFromImages(data_path, label_map, self.n_to_n, model_config["sequence_length"],
                                                 model_config["grayscale"], image_sizes, transform=data_transforms,
                                                 limit=limit, defects=defects, load_data=load_data)
            else:
                self.dataset = PytorchVideoDataset(data_path, label_map, self.n_to_n, model_config["sequence_length"],
                                                   model_config["grayscale"], image_sizes, transform=data_transforms,
                                                   limit=limit, load_data=load_data)

            self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                          batch_size=batch_size,
                                                          shuffle=(mode == "Train"),
                                                          num_workers=num_workers,
                                                          drop_last=drop_last)

        msg = f"{mode} data loaded"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)))

    def __iter__(self):
        return DataIterator(self)

    def __len__(self):
        return len(self.dataset)


class DataIterator(object):
    """An iterator."""
    def __init__(self, dataloader: VideoDataloader):
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader.dataloader)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # TODO: remove the hardcoded 0 ?

    def __iter__(self):
        return self

    def __next__(self):
        # DALI has an extra dimension (don't know why), and the data is already on GPU.
        batch = next(self.dataloader_iter)
        if self.dataloader.dali:
            pass
            # batch = batch[0]
            # # Fix for n_to_n mode. Should not be done for n to 1 mode.
            # if self.dataloader.n_to_n:
            #     batch["label"] = batch["label"].repeat(*batch["data"].shape[:2]).long()
            # # print(f"DALI labels: {batch['label']}")
            # return batch
        else:
            batch["data"] = batch["data"].to(self.device).float()
            batch["label"] = batch["label"].to(self.device).long()
            return batch

    def __len__(self):
        return len(self.dataloader)
