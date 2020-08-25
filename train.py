from torch.utils.tensorboard import SummaryWriter  # noqa: F401  # Needs to be there to avoid segfaults
import os
import glob
import shutil
import time

import torch
import torchvision.transforms as transforms
from torchsummary import summary

from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.dataset.dataset import Dataset
from src.networks.network import Network
from src.train import train


def main():
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = Dataset(os.path.join(DataConfig.DATA_PATH, "Train"),
                            transform=transforms.Compose([
                                transforms.Resize((ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE)),
                                transforms.Grayscale(),
                                # transforms.RandomHorizontalFlip(),   # needs to be same flip for whole video
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, ), (0.5, ))
                            ]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=ModelConfig.BATCH_SIZE,
                                                   shuffle=True, num_workers=ModelConfig.WORKERS)

    print("Train data loaded" + ' ' * (os.get_terminal_size()[0] - 17))

    val_dataset = Dataset(os.path.join(DataConfig.DATA_PATH, "Validation"),
                          transform=transforms.Compose([
                              transforms.Resize((ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE)),
                              transforms.Grayscale(),
                              # transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, ), (0.5, ))
                          ]))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=ModelConfig.BATCH_SIZE,
                                                 shuffle=False, num_workers=ModelConfig.WORKERS)
    print("Validation data loaded" + ' ' * (os.get_terminal_size()[0] - 22))

    print(f"\nLoaded {len(train_dataloader.dataset)} train data and",
          f"{len(val_dataloader.dataset)} validation data", flush=True)

    model = Network()
    model = model.float()
    model.to(device)
    # The summary does not work with an LSTM for some reason
    # summary(model, (ModelConfig.VIDEO_SIZE, 1, ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE))

    train(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
