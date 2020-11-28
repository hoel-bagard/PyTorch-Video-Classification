import os
import time

import torch

from config.model_config import ModelConfig


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
        self.train_steps_per_epoch = (len(train_dataloader) + (batch_size - 1)) // batch_size
        self.val_steps_per_epoch = (len(val_dataloader) + (batch_size - 1)) // batch_size

    def train_epoch(self):
        epoch_loss = 0.0
        step_time, fetch_time = None, None
        step_start_time = time.perf_counter()  # Needs to be outside the loop to include dataloading
        for step, batch in enumerate(self.train_dataloader, start=1):
            data_loading_finished_time = time.perf_counter()
            self.optimizer.zero_grad()

            inputs, labels = batch["video"], batch["label"]

            if ModelConfig.NETWORK == "LRCN":
                self.model.reset_lstm_state(inputs.shape[0])

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            previous_step_start_time = step_start_time
            if step_time:
                step_time = 0.9*step_time + 0.1*1000*(time.perf_counter() - step_start_time)
                fetch_time = 0.9*fetch_time + 0.1*1000*(data_loading_finished_time - previous_step_start_time)
            else:
                step_time = 1000*(time.perf_counter() - step_start_time)
                fetch_time = 1000*(data_loading_finished_time - previous_step_start_time)
            step_start_time = time.perf_counter()
            self._print(step, self.train_steps_per_epoch, loss, step_time, fetch_time)

        return epoch_loss / self.train_steps_per_epoch

    def val_epoch(self):
        epoch_loss = 0.0
        step_time, fetch_time = None, None
        step_start_time = time.perf_counter()  # Needs to be outside the loop to include dataloading
        for step, batch in enumerate(self.val_dataloader, start=1):
            data_loading_finished_time = time.perf_counter()

            inputs, labels = batch["video"], batch["label"]

            if ModelConfig.NETWORK == "LRCN":
                self.model.reset_lstm_state(inputs.shape[0])

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            epoch_loss += loss.item()

            previous_step_start_time = step_start_time
            if step_time:
                step_time = 0.9*step_time + 0.1*1000*(time.perf_counter() - step_start_time)
                fetch_time = 0.9*fetch_time + 0.1*1000*(data_loading_finished_time - previous_step_start_time)
            else:
                step_time = 1000*(time.perf_counter() - step_start_time)
                fetch_time = 1000*(data_loading_finished_time - previous_step_start_time)
            step_start_time = time.perf_counter()
            self._print(step, self.val_steps_per_epoch, loss, step_time, fetch_time)

        return epoch_loss / self.val_steps_per_epoch

    @staticmethod
    def _print(step, max_steps, loss, step_time, fetch_time):
        pre_string = f"{step}/{max_steps} ["
        post_string = (f"],  Loss: {loss.item():.3e}  -  Step time: {step_time:.2f}ms"
                       f"  -  Fetch time: {fetch_time:.2f}ms    ")
        terminal_cols = os.get_terminal_size().columns
        progress_bar_len = min(terminal_cols - len(pre_string) - len(post_string)-1, 30)
        epoch_progress = int(progress_bar_len * (step/max_steps))
        print(pre_string + f"{epoch_progress*'='}>{(progress_bar_len-epoch_progress)*'.'}" + post_string,
              end='\r', flush=True)
