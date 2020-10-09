import time
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

from config.model_config import ModelConfig
from config.data_config import DataConfig
from src.utils.trainer import Trainer
from src.utils.tensorboard import TensorBoard
from src.utils.metrics import Metrics


def train(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader):
    loss_fn = nn.NLLLoss()  # nn.CrossEntropyLoss()
    trainer = Trainer(model, loss_fn, train_dataloader, val_dataloader)
    scheduler = ExponentialLR(trainer.optimizer, gamma=ModelConfig.LR_DECAY)
    if DataConfig.USE_TB:
        metrics = Metrics(model, loss_fn, train_dataloader, val_dataloader)
        tensorboard = TensorBoard(model, metrics)

    best_loss = 1000
    last_checkpoint_epoch = 0

    for epoch in range(ModelConfig.MAX_EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch}/{ModelConfig.MAX_EPOCHS}")

        epoch_loss = trainer.train_epoch()
        if DataConfig.USE_TB:
            tensorboard.write_loss(epoch, epoch_loss)
            tensorboard.write_lr(epoch, scheduler.get_last_lr()[0])

        if (epoch_loss < best_loss and DataConfig.USE_CHECKPOINT and
                epoch >= DataConfig.RECORD_START and (epoch - last_checkpoint_epoch) >= DataConfig.CHECKPT_SAVE_FREQ):
            save_path = os.path.join(DataConfig.CHECKPOINT_DIR, f"train_{epoch}.pt")
            print(f"\nLoss improved from {best_loss:.5e} to {epoch_loss:.5e}, saving model to {save_path}", end='\r')
            best_loss, last_checkpoint_epoch = epoch_loss, epoch
            torch.save(model.state_dict(), save_path)

        print(f"\nEpoch loss: {epoch_loss:.5e}  -  Took {time.time() - epoch_start_time:.5f}s")

        # Validation and other metrics
        if epoch % DataConfig.VAL_FREQ == 0 and epoch >= DataConfig.RECORD_START:
            with torch.no_grad():
                validation_start_time = time.time()
                epoch_loss = trainer.val_epoch()

                if DataConfig.USE_TB:
                    print("\nStarting to compute TensorBoard metrics", end="\r", flush=True)
                    tensorboard.write_loss(epoch, epoch_loss, mode="Validation")

                    # Metrics for the Train dataset
                    tensorboard.write_images(epoch, train_dataloader)
                    # tensorboard.write_videos(epoch, train_dataloader)
                    train_acc = tensorboard.write_metrics(epoch)

                    # Metrics for the Validation dataset
                    tensorboard.write_images(epoch, val_dataloader, mode="Validation")
                    if epoch % (3*DataConfig.VAL_FREQ) == 0:
                        tensorboard.write_videos(epoch, val_dataloader, mode="Validation")
                    val_acc = tensorboard.write_metrics(epoch, mode="Validation")

                    print(f"\nTrain accuracy: {train_acc:.3f}  -  Validation accuracy: {val_acc:.3f}",
                          end='\r', flush=True)

                print(f"\nValidation loss: {epoch_loss:.5e}  -  Took {time.time() - validation_start_time:.5f}s",
                      flush=True)
        scheduler.step()

    print("Finished Training")
