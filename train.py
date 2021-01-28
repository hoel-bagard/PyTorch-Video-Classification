from torch.utils.tensorboard import SummaryWriter  # noqa: F401  # Needs to be there to avoid segfaults
import argparse
import os
import glob
import shutil
import time

import torch
from torchsummary import summary

from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.utils.config_to_kwargs import get_model_config_dict
from src.torch_utils.dataset.build_video_dataloader import VideoDataloader
from src.networks.build_network import build_model
from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", "-l", default=None, type=int, help="Limits the number of apparition of each class")
    parser.add_argument("--load_data", "-ld", action="store_true", help="Loads all the videos into RAM")
    args = parser.parse_args()

    if not DataConfig.KEEP_TB:
        while os.path.exists(DataConfig.TB_DIR):
            shutil.rmtree(DataConfig.TB_DIR, ignore_errors=True)
            time.sleep(0.5)
    os.makedirs(DataConfig.TB_DIR, exist_ok=True)

    if DataConfig.USE_CHECKPOINT:
        if not DataConfig.KEEP_CHECKPOINTS:
            while os.path.exists(DataConfig.CHECKPOINT_DIR):
                shutil.rmtree(DataConfig.CHECKPOINT_DIR, ignore_errors=True)
                time.sleep(0.5)
        try:
            os.makedirs(DataConfig.CHECKPOINT_DIR, exist_ok=False)
        except FileExistsError:
            print(f"The checkpoint dir {DataConfig.CHECKPOINT_DIR} already exists")
            return -1

        # Makes a copy of all the code (and config) so that the checkpoints are easy to load and use
        output_folder = os.path.join(DataConfig.CHECKPOINT_DIR, "Classification-PyTorch")
        for filepath in glob.glob(os.path.join("**", "*.py"), recursive=True):
            destination_path = os.path.join(output_folder, filepath)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy(filepath, destination_path)
        # shutil.copytree(".git", os.path.join(output_folder, ".git"))
        misc_files = ["README.md", "requirements.txt", "setup.cfg", ".gitignore"]
        for misc_file in misc_files:
            shutil.copy(misc_file, os.path.join(output_folder, misc_file))
        print("Finished copying files")

    torch.backends.cudnn.benchmark = True   # Makes training quite a bit faster

    train_dataloader = VideoDataloader(os.path.join(DataConfig.DATA_PATH, "Train"), DataConfig.DALI,
                                       DataConfig.LABEL_MAP, drop_last=ModelConfig.MODEL.__name__ == "LRCN",
                                       num_workers=DataConfig.NUM_WORKERS, dali_device_id=DataConfig.DALI_DEVICE_ID,
                                       limit=args.limit, **get_model_config_dict())

    val_dataloader = VideoDataloader(os.path.join(DataConfig.DATA_PATH, "Validation"), DataConfig.DALI,
                                     DataConfig.LABEL_MAP, drop_last=ModelConfig.MODEL.__name__ == "LRCN",
                                     num_workers=DataConfig.NUM_WORKERS, dali_device_id=DataConfig.DALI_DEVICE_ID,
                                     limit=args.limit, **get_model_config_dict())

    print(f"Loaded {len(train_dataloader)} train data and", f"{len(val_dataloader)} validation data", flush=True)
    print("Building model. . .", end="\r")

    model = build_model(ModelConfig.MODEL, DataConfig.OUTPUT_CLASSES, **get_model_config_dict())
    # The summary does not work with an LSTM for some reason
    if ModelConfig.MODEL.__name__ != "LRCN":
        summary(model, (ModelConfig.SEQUENCE_LENGTH, 1 if ModelConfig.GRAYSCALE else 3,
                        ModelConfig.IMAGE_SIZES[0], ModelConfig.IMAGE_SIZES[1]))

    train(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
