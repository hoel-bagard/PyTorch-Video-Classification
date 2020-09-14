import time
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
# import numpy as np

from config.model_config import ModelConfig
from config.data_config import DataConfig
from src.utils.trainer import Trainer
from src.utils.draw import draw_pred
from src.utils.accuracy import get_accuracy


def train(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.NLLLoss()  # nn.CrossEntropyLoss()
    trainer = Trainer(model, loss_fn, train_dataloader, val_dataloader)
    scheduler = ExponentialLR(trainer.optimizer, gamma=ModelConfig.LR_DECAY)
    tb_writer = SummaryWriter(DataConfig.TB_DIR)
    # tb_writer.add_graph(model, (torch.empty(1, ModelConfig.VIDEO_SIZE,
    #                                         1, ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE, device=device), ))
    tb_writer.flush()

    best_loss = 1000
    last_checkpoint_epoch = 0

    for epoch in range(ModelConfig.MAX_EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch}/{ModelConfig.MAX_EPOCHS}")

        epoch_loss = trainer.train_epoch()
        if DataConfig.USE_TB:
            tb_writer.add_scalar("Training Loss", epoch_loss, epoch)
            tb_writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)
            tb_writer.flush()

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
                    tb_writer.add_scalar('Validation loss', epoch_loss, epoch)

                    # Metrics for the Train dataset
                    batch = next(iter(train_dataloader))
                    video, labels = batch["video"].float(), batch["label"]
                    if ModelConfig.MODEL == "LRCN":
                        model.reset_lstm_state(video.shape[0])
                    predictions = model(video.to(device))
                    predictions = torch.nn.functional.softmax(predictions, dim=-1)
                    train_acc = get_accuracy(labels, predictions.cpu())
                    tb_writer.add_scalar("Training Accuracy", train_acc, epoch)
                    # out_imgs = draw_pred(in_imgs, predictions, labels)
                    # for image_index, out_img in enumerate(out_imgs):
                    #     out_img = np.transpose(out_img, (2, 0, 1))  # HWC -> CHW
                    #     tb_writer.add_image(f"Train/prediction_{image_index}", out_img, global_step=epoch)

                    # Metrics for the Validation dataset
                    batch = next(iter(val_dataloader))
                    video, labels = batch["video"].float(), batch["label"]
                    if ModelConfig.MODEL == "LRCN":
                        model.reset_lstm_state(video.shape[0])
                    predictions = model(video.to(device))
                    predictions = torch.nn.functional.softmax(predictions, dim=-1)
                    val_acc = get_accuracy(labels, predictions.cpu())
                    tb_writer.add_scalar("Validation Accuracy", val_acc, epoch)
                    # out_imgs = draw_pred(in_imgs, predictions, labels)
                    # for image_index, out_img in enumerate(out_imgs):
                    #     out_img = np.transpose(out_img, (2, 0, 1))  # HWC -> CHW
                    #     tb_writer.add_image(f"Validation/prediction_{image_index}", out_img, global_step=epoch)
                    print(f"\nTrain accuracy: {train_acc:.3f}  -  Validation accuracy: {val_acc:.3f}", end='\r', flush=True)

                print(f"\nValidation loss: {epoch_loss:.5e}  -  Took {time.time() - validation_start_time:.5f}s", flush=1)
        scheduler.step()

    print("Finished Training")
