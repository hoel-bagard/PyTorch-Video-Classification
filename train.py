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
from src.dataset.build_dataloader import Dataloader
from src.networks.build_network import build_model
from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", default=None, type=int, help="Limits the number of apparition of each class")
    parser.add_argument("--load_data", action="store_true", help="Loads all the videos into RAM")
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

    train_dataloader = Dataloader(os.path.join(DataConfig.DATA_PATH, "Train"),
                                  limit=args.limit)

    val_dataloader = Dataloader(os.path.join(DataConfig.DATA_PATH, "Validation"),
                                limit=args.limit)

    print(f"\nLoaded {len(train_dataloader)} train data and",
          f"{len(val_dataloader)} validation data", flush=True)

    model = build_model(ModelConfig.NETWORK)
    # The summary does not work with an LSTM for some reason
    if ModelConfig.NETWORK != "LRCN":
        summary(model, (ModelConfig.VIDEO_SIZE, 1 if ModelConfig.USE_GRAY_SCALE else 3,
                        ModelConfig.IMAGE_SIZES[0], ModelConfig.IMAGE_SIZES[1]))

    train(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
