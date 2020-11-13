import time

import torch

from config.model_config import ModelConfig
from config.data_config import DataConfig


class Trainer:
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(model.parameters(), lr=ModelConfig.LR, weight_decay=ModelConfig.REG_FACTOR)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        batch_size = ModelConfig.BATCH_SIZE
        if DataConfig.DALI:
            self.train_steps_per_epoch = (len(train_dataloader) + (batch_size - 1)) // batch_size
            self.val_steps_per_epoch = (len(val_dataloader) + (batch_size - 1)) // batch_size
        else:
            self.train_steps_per_epoch = (len(train_dataloader.dataset) + (batch_size - 1)) // batch_size
            self.val_steps_per_epoch = (len(val_dataloader.dataset) + (batch_size - 1)) // batch_size

    def train_epoch(self):
        epoch_loss = 0.0
        step_start_time = time.perf_counter()  # Needs to be outside the loop to include dataloading
        for step, batch in enumerate(self.train_dataloader, start=1):
            self.optimizer.zero_grad()

            if DataConfig.DALI:
                inputs, labels = batch[0]["video"].float(), batch[0]["label"].long()
            else:
                inputs, labels = batch["video"].to(self.device).float(), batch["label"].to(self.device).long()

            if ModelConfig.NETWORK == "LRCN":
                self.model.reset_lstm_state(inputs.shape[0])

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_progress = int(30 * (step/self.train_steps_per_epoch))
            print(f"{step}/{self.train_steps_per_epoch} [" + epoch_progress*"=" + ">" + (30-epoch_progress)*"." + "]",
                  f",  Loss: {loss.item():.3e}",
                  f"  -  Step time: {1000*(time.perf_counter() - step_start_time):.2f}ms    ",
                  end='\r', flush=True)
            epoch_loss += loss.item()
            step_start_time = time.perf_counter()
        return epoch_loss / self.train_steps_per_epoch

    def val_epoch(self):
        epoch_loss = 0.0
        step_start_time = time.perf_counter()  # Needs to be outside the loop to include dataloading
        for step, batch in enumerate(self.val_dataloader, start=1):

            if DataConfig.DALI:
                inputs, labels = batch[0]["video"].float(), batch[0]["label"].long()
            else:
                inputs, labels = batch["video"].to(self.device).float(), batch["label"].to(self.device).long()

            if ModelConfig.NETWORK == "LRCN":
                self.model.reset_lstm_state(inputs.shape[0])

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)

            epoch_progress = int(30 * (step/self.val_steps_per_epoch))
            print(f"{step}/{self.val_steps_per_epoch} [" + epoch_progress*"=" + ">" + (30-epoch_progress)*"." + "]",
                  f",  Loss: {loss.item():.3e}",
                  f"  -  Step time: {1000*(time.time() - step_start_time):.2f}ms    ",
                  end='\r', flush=True)
            epoch_loss += loss.item()
            step_start_time = time.time()
        return epoch_loss / self.val_steps_per_epoch
