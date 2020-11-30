import time
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

from .torch_utils.config_templates.data_config_template import DataConfig
from .torch_utils.config_templates.model_config_template import ModelConfig
from src.utils.trainer import Trainer
from src.utils.tensorboard import TensorBoard
from src.utils.metrics import Metrics
from src.losses import CE_Loss


def train(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader,
          data_config: DataConfig, model_config: ModelConfig):
    if model_config.N_TO_N:
        loss_fn = CE_Loss()
    else:
        loss_fn = nn.CrossEntropyLoss()  # nn.NLLLoss()
    trainer = Trainer(model, loss_fn, train_dataloader, val_dataloader)
    scheduler = ExponentialLR(trainer.optimizer, gamma=model_config.LR_DECAY)
    if data_config.USE_TB:
        metrics = Metrics(model, loss_fn, train_dataloader, val_dataloader, model_config.OUTPUT_CLASSES, max_batches=None)
        tensorboard = TensorBoard(model, metrics)

    best_loss = 1000
    last_checkpoint_epoch = 0

    for epoch in range(model_config.MAX_EPOCHS):
        epoch_start_time = time.perf_counter()
        print(f"\nEpoch {epoch}/{model_config.MAX_EPOCHS}")

        epoch_loss = trainer.train_epoch()
        if data_config.USE_TB:
            tensorboard.write_loss(epoch, epoch_loss)
            tensorboard.write_lr(epoch, scheduler.get_last_lr()[0])

        if (epoch_loss < best_loss and data_config.USE_CHECKPOINT and
                epoch >= data_config.RECORD_START and (epoch - last_checkpoint_epoch) >= data_config.CHECKPT_SAVE_FREQ):
            save_path = os.path.join(data_config.CHECKPOINT_DIR, f"train_{epoch}.pt")
            print(f"\nLoss improved from {best_loss:.5e} to {epoch_loss:.5e}, saving model to {save_path}", end='\r')
            best_loss, last_checkpoint_epoch = epoch_loss, epoch
            torch.save(model.state_dict(), save_path)

        print(f"\nEpoch loss: {epoch_loss:.5e}  -  Took {time.perf_counter() - epoch_start_time:.5f}s")

        # Validation and other metrics
        if epoch % data_config.VAL_FREQ == 0 and epoch >= data_config.RECORD_START:
            with torch.no_grad():
                validation_start_time = time.perf_counter()
                epoch_loss = trainer.val_epoch()

                if data_config.USE_TB:
                    print("\nStarting to compute TensorBoard metrics", end="\r", flush=True)
                    tensorboard.write_loss(epoch, epoch_loss, mode="Validation")

                    # Metrics for the Train dataset
                    tensorboard.write_images(epoch, train_dataloader)
                    if epoch % (3*data_config.VAL_FREQ) == 0:
                        tensorboard.write_videos(epoch, train_dataloader)
                    train_acc = tensorboard.write_metrics(epoch)

                    # Metrics for the Validation dataset
                    tensorboard.write_images(epoch, val_dataloader, mode="Validation")
                    if epoch % (3*data_config.VAL_FREQ) == 0:
                        tensorboard.write_videos(epoch, val_dataloader, mode="Validation")
                    val_acc = tensorboard.write_metrics(epoch, mode="Validation")

                    print(f"\nTrain accuracy: {train_acc:.3f}  -  Validation accuracy: {val_acc:.3f}",
                          end='\r', flush=True)

                print(f"\nValidation loss: {epoch_loss:.5e}  -"
                      f"  Took {time.perf_counter() - validation_start_time:.5f}s",
                      flush=True)
        scheduler.step()

    print("Finished Training")
