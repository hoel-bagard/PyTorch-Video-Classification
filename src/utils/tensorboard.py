import os

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.utils.draw import (
    draw_pred,
    draw_pred_video
)
from src.utils.metrics import Metrics


class TensorBoard():
    def __init__(self, model: nn.Module, metrics: Metrics, max_outputs: int = 4):
        """
        Args:
            model: Model'whose performance are to be recorded
            max_outputs: Number of images kept and dislpayed in TensorBoard
        """
        super(TensorBoard, self).__init__()
        self.max_outputs = max_outputs
        self.metrics: Metrics = metrics
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_tb_writer = SummaryWriter(os.path.join(DataConfig.TB_DIR, "Train"))
        self.val_tb_writer = SummaryWriter(os.path.join(DataConfig.TB_DIR, "Validation"))
        if ModelConfig.NETWORK != "LRCN":
            self.train_tb_writer.add_graph(model, (torch.empty(2, ModelConfig.VIDEO_SIZE,
                                                   1 if ModelConfig.USE_GRAY_SCALE else 3,
                                                   ModelConfig.IMAGE_SIZES[0], ModelConfig.IMAGE_SIZES[1],
                                                   device=self.device), ))
            self.train_tb_writer.flush()

    def write_images(self, epoch: int, dataloader: torch.utils.data.DataLoader, mode: str = "Train"):
        """
        Writes images with predictions written on them to TensorBoard
        Args:
            epoch: Current epoch
            dataloader: The images will be sampled from this dataset
            mode: Either "Train" or "Validation"
        """
        print("Writing images" + ' ' * (os.get_terminal_size()[0] - len("Writing images")), end="\r", flush=True)
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        # Get some data
        batch = next(iter(dataloader))
        if ModelConfig.NETWORK == "LRCN":  # LSTM needs proper batches (the pytorch implementation at least)
            videos, labels = batch["video"].float(), batch["label"][:self.max_outputs]
            self.model.reset_lstm_state(videos.shape[0])
        else:
            videos, labels = batch["video"][:self.max_outputs].float(), batch["label"][:self.max_outputs]

        # Get some predictions
        predictions = self.model(videos.to(self.device))
        if ModelConfig.NETWORK == "LRCN":
            predictions, videos = predictions[:self.max_outputs], videos[:self.max_outputs]
        predictions = torch.nn.functional.softmax(predictions, dim=-1)

        # Write prediction on some images and add them to TensorBoard
        out_imgs = draw_pred(videos, predictions, labels)

        for image_index, out_img in enumerate(out_imgs):
            # If opencv resizes the image, it removes the channel dimension
            if out_img.ndim == 2:
                out_img = np.expand_dims(out_img, -1)
            out_img = rearrange(out_img, 'w h c -> c w h')
            tb_writer.add_image(f"{mode}/prediction_{image_index}", out_img, global_step=epoch)

    def write_videos(self, epoch: int, dataloader: torch.utils.data.DataLoader, mode: str = "Train"):
        """
        Write a video with predictions written on it to TensorBoard
        Args:
            epoch: Current epoch
            dataloader: The images will be sampled from this dataset
            mode: Either "Train" or "Validation"
        """
        print("Writing videos" + ' ' * (os.get_terminal_size()[0] - len("Writing videos")), end="\r", flush=True)
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        # Get some data
        batch = next(iter(dataloader))
        if ModelConfig.NETWORK == "LRCN":  # LSTM needs proper batches (the pytorch implementation at least)
            videos, labels = batch["video"].float(), batch["label"][:self.max_outputs]
            self.model.reset_lstm_state(videos.shape[0])
        else:
            videos, labels = batch["video"][:1].float(), batch["label"][:1]

        # Get some predictions
        predictions = self.model(videos.to(self.device))
        if ModelConfig.NETWORK == "LRCN":
            predictions, videos = predictions[:1], videos[:1]
        predictions = torch.nn.functional.softmax(predictions, dim=-1)

        # Write prediction on a video and add it to TensorBoard
        out_video = draw_pred_video(videos[0], predictions[0], labels[0])
        out_video = np.transpose(out_video, (0, 3, 1, 2))  # HWC -> CHW
        out_video = np.expand_dims(out_video, 0)  # Re-add batch dimension

        tb_writer.add_video("Video", out_video, global_step=epoch, fps=16)

    def write_metrics(self, epoch: int, mode: str = "Train") -> float:
        """
        Writes accuracy metrics in TensorBoard
        Args:
            epoch: Current epoch
            mode: Either "Train" or "Validation"
        Returns:
            avg_acc: Average accuracy
        """
        print("Computing confusion matrix" + ' ' * (os.get_terminal_size()[0] - 26), end="\r", flush=True)
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        self.metrics.compute_confusion_matrix(mode=mode)

        print("Computing average accuracy" + ' ' * (os.get_terminal_size()[0] - 26), end="\r", flush=True)
        avg_acc = self.metrics.get_avg_acc()
        tb_writer.add_scalar("Average Accuracy", avg_acc, epoch)

        print("Computing per class accuracy" + ' ' * (os.get_terminal_size()[0] - 28), end="\r", flush=True)
        per_class_acc = self.metrics.get_class_accuracy()
        for key, acc in enumerate(per_class_acc):
            tb_writer.add_scalar(f"Per Class Accuracy/{DataConfig.LABEL_MAP[key]}", acc, epoch)

        print("Creating confusion matrix image" + ' ' * (os.get_terminal_size()[0] - 31), end="\r", flush=True)
        confusion_matrix = self.metrics.get_confusion_matrix()
        confusion_matrix = np.transpose(confusion_matrix, (2, 0, 1))  # HWC -> CHW
        tb_writer.add_image("Confusion Matrix", confusion_matrix, global_step=epoch)

        return avg_acc

    def write_loss(self, epoch: int, loss: float, mode: str = "Train"):
        """
        Writes loss metric in TensorBoard
        Args:
            epoch: Current epoch
            loss: Epoch loss that will be added to the TensorBoard
            mode: Either "Train" or "Validation"
        """
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        tb_writer.add_scalar("Loss", loss, epoch)
        self.train_tb_writer.flush()

    def write_lr(self, epoch: int, lr: float):
        """
        Writes learning rate in the TensorBoard
        Args:
            epoch: Current epoch
            lr: Learning rate for the given epoch
        """
        self.train_tb_writer.add_scalar("Learning Rate", lr, epoch)
