import time

import torch

from config.model_config import ModelConfig
from src.networks.network import LRCN


class Trainer:
    def __init__(self, model: LRCN, loss_fn,
                 train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader):
        self.model: LRCN = model
        self.loss_fn = loss_fn
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(),
                                                                 lr=ModelConfig.LR, weight_decay=ModelConfig.REG_FACTOR)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        batch_size = ModelConfig.BATCH_SIZE
        self.train_steps_per_epoch = (len(train_dataloader.dataset) + (batch_size - 1)) // batch_size
        self.val_steps_per_epoch = (len(val_dataloader.dataset) + (batch_size - 1)) // batch_size

    def train_epoch(self):
        epoch_loss = 0.0
        for step, batch in enumerate(self.train_dataloader, start=1):
            step_start_time = time.time()
            self.optimizer.zero_grad()

            inputs, labels = batch["video"].to(self.device).float(), batch["label"].to(self.device).long()
            if ModelConfig.MODEL == "LRCN":
                self.model.reset_lstm_state(inputs.shape[0])

            outputs = self.model(inputs)

            # If predicting for every frame
            # labels = labels.unsqueeze(-1)
            # labels = labels * torch.ones(outputs.size()[:-1], device=self.device).long()
            # outputs = outputs.permute((0, 2, 1))

            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_progress = int(30 * (step/self.train_steps_per_epoch))
            print(f"{step}/{self.train_steps_per_epoch} [" + epoch_progress*"=" + ">" + (30-epoch_progress)*"." + "]",
                  f",  Loss: {loss.item():.3e}",
                  f"  -  Step time: {1000*(time.time() - step_start_time):.2f}ms    ",
                  end='\r', flush=True)
            epoch_loss += loss.item()
        return epoch_loss / self.train_steps_per_epoch

    def val_epoch(self):
        epoch_loss = 0.0
        for step, batch in enumerate(self.val_dataloader, start=1):
            step_start_time = time.time()

            inputs, labels = batch["video"].to(self.device).float(), batch["label"].to(self.device).long()
            if ModelConfig.MODEL == "LRCN":
                self.model.reset_lstm_state(inputs.shape[0])

            outputs = self.model(inputs)

            # If predicting for every frame
            # labels = labels.unsqueeze(-1)
            # labels = labels * torch.ones(outputs.size()[:-1], device=self.device).long()
            # outputs = outputs.permute((0, 2, 1))

            loss = self.loss_fn(outputs, labels)

            epoch_progress = int(30 * (step/self.val_steps_per_epoch))
            print(f"{step}/{self.val_steps_per_epoch} [" + epoch_progress*"=" + ">" + (30-epoch_progress)*"." + "]",
                  f",  Loss: {loss.item():.3e}",
                  f"  -  Step time: {1000*(time.time() - step_start_time):.2f}ms    ",
                  end='\r', flush=True)
            epoch_loss += loss.item()
        return epoch_loss / self.val_steps_per_epoch
