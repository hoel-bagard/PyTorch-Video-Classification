import time
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

from config.model_config import ModelConfig
from config.data_config import DataConfig
from src.torch_utils.utils.trainer import Trainer
from src.torch_utils.utils.tensorboard import TensorBoard
from src.torch_utils.utils.metrics import Metrics
from src.torch_utils.utils.ressource_usage import resource_usage
from src.losses import CE_Loss


def train(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader):
    if ModelConfig.N_TO_N:
        loss_fn = CE_Loss()
    else:
        loss_fn = nn.CrossEntropyLoss()  # nn.NLLLoss()

    on_epoch_begin = model.reset_lstm_state if model.__class__.__name__ == "LRCN" else None
    trainer = Trainer(model, loss_fn, train_dataloader, val_dataloader, ModelConfig.BATCH_SIZE, ModelConfig.LR,
                      ModelConfig.REG_FACTOR, on_epoch_begin=on_epoch_begin)
    scheduler = ExponentialLR(trainer.optimizer, gamma=ModelConfig.LR_DECAY)
    if DataConfig.USE_TB:
        metrics = Metrics(model, loss_fn, train_dataloader, val_dataloader,
                          DataConfig.LABEL_MAP, ModelConfig.N_TO_N, max_batches=None)
        tensorboard = TensorBoard(model, metrics, DataConfig.LABEL_MAP, DataConfig.TB_DIR, ModelConfig.SEQUENCE_LENGTH,
                                  ModelConfig.N_TO_N, ModelConfig.GRAYSCALE, ModelConfig.IMAGE_SIZES)

    best_loss = 1000
    last_checkpoint_epoch = 0
    train_start_time = time.time()

    preprocess_fn = getattr(model, "preprocess", None)
    if not callable(preprocess_fn):
        preprocess_fn = None
    postprocess_fn = getattr(model, "postprocess", None)
    if not callable(postprocess_fn):
        postprocess_fn = None

    try:
        for epoch in range(ModelConfig.MAX_EPOCHS):
            epoch_start_time = time.perf_counter()
            print(f"\nEpoch {epoch}/{ModelConfig.MAX_EPOCHS}")

            epoch_loss = trainer.train_epoch()
            if DataConfig.USE_TB:
                tensorboard.write_loss(epoch, epoch_loss)
                tensorboard.write_lr(epoch, scheduler.get_last_lr()[0])

            if (epoch_loss < best_loss and DataConfig.USE_CHECKPOINT and epoch >= DataConfig.RECORD_START
                    and (epoch - last_checkpoint_epoch) >= DataConfig.CHECKPT_SAVE_FREQ):
                save_path = os.path.join(DataConfig.CHECKPOINT_DIR, f"train_{epoch}.pt")
                print(f"\nLoss improved from {best_loss:.5e} to {epoch_loss:.5e},"
                      f"saving model to {save_path}", end='\r')
                best_loss, last_checkpoint_epoch = epoch_loss, epoch
                torch.save(model.state_dict(), save_path)

            print(f"\nEpoch loss: {epoch_loss:.5e}  -  Took {time.perf_counter() - epoch_start_time:.5f}s")

            # Validation and other metrics
            if epoch % DataConfig.VAL_FREQ == 0 and epoch >= DataConfig.RECORD_START:
                with torch.no_grad():
                    validation_start_time = time.perf_counter()
                    epoch_loss = trainer.val_epoch()

                    if DataConfig.USE_TB:
                        print("\nStarting to compute TensorBoard metrics", end="\r", flush=True)
                        tensorboard.write_loss(epoch, epoch_loss, mode="Validation")

                        # Metrics for the Train dataset
                        tensorboard.write_images(epoch, train_dataloader, input_is_video=True,
                                                 preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
                        if epoch % (3*DataConfig.VAL_FREQ) == 0:
                            tensorboard.write_videos(epoch, train_dataloader,
                                                     preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
                        train_acc = tensorboard.write_metrics(epoch)

                        # Metrics for the Validation dataset
                        tensorboard.write_images(epoch, val_dataloader, mode="Validation", input_is_video=True,
                                                 preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
                        if epoch % (3*DataConfig.VAL_FREQ) == 0:
                            tensorboard.write_videos(epoch, val_dataloader, mode="Validation",
                                                     preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
                        val_acc = tensorboard.write_metrics(epoch, mode="Validation")

                        print(f"Train accuracy: {train_acc:.3f}  -  Validation accuracy: {val_acc:.3f}",
                              end='\r', flush=True)

                    print(f"\nValidation loss: {epoch_loss:.5e}  -"
                          f"  Took {time.perf_counter() - validation_start_time:.5f}s",
                          flush=True)
            scheduler.step()
    except KeyboardInterrupt:
        print("\n")

    train_stop_time = time.time()
    tensorboard.close_writers()
    memory_peak, gpu_memory = resource_usage()
    print("Finished Training"
          f"\n\tTraining time : {train_stop_time - train_start_time:.03f}s"
          f"\n\tRAM peak : {memory_peak // 1024} MB\n\tVRAM usage : {gpu_memory}")
