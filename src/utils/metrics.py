import itertools
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.data_config import DataConfig


class Metrics:
    def __init__(self, model: nn.Module, loss_fn, train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader, max_batches: int = 10):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_batches = max_batches

    def compute_confusion_matrix(self, mode: str = "Train"):
        """
        Computes the confusion matrix. This function has to be called before using the get functions.
        Args:
            mode: Either "Train" or "Validation"
        """
        self.cm = np.zeros((ModelConfig.OUTPUT_CLASSES, ModelConfig.OUTPUT_CLASSES))
        for step, batch in enumerate(self.train_dataloader if mode == "Train" else self.val_dataloader, start=1):
            imgs, labels_batch = batch["video"].float(), batch["label"].cpu().detach().numpy()
            predictions_batch = self.model(imgs.to(self.device))
            predictions_batch = torch.nn.functional.softmax(predictions_batch, dim=-1)
            predictions_batch = torch.argmax(predictions_batch, dim=-1).int().cpu().detach().numpy()

            for (label_video, pred_video) in zip(labels_batch, predictions_batch):  # batch
                if ModelConfig.USE_N_TO_N:
                    for (label_frame, pred_frame) in zip(label_video, pred_video):
                        self.cm[label_frame, pred_frame] += 1
                else:
                    self.cm[label_video, pred_video] += 1

            if self.max_batches and step >= self.max_batches:
                break

    def get_avg_acc(self) -> float:
        """
        Uses the confusion matrix to return the average accuracy of the model
        Returns:
            avg_acc: Average accuracy
        """
        avg_acc = np.sum([self.cm[i, i] for i in range(len(self.cm))]) / np.sum(self.cm)
        return avg_acc

    def get_class_accuracy(self) -> List[float]:
        """
        Uses the confusion matrix to return the average accuracy of the model
        Returns:
            per_class_acc: An array containing the accuracy for each class
        """
        per_class_acc = [self.cm[i, i] / np.sum(self.cm[i]) for i in range(len(self.cm))]
        return per_class_acc

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Returns an image containing the plotted confusion matrix.
        Taken from: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12

        Returns:
            per_class_acc: Image of the confusion matrix.
        """
        cm = self.cm
        class_names = DataConfig.LABEL_MAP.values()

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Normalize the confusion matrix.
        cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel("True label", labelpad=-5)
        plt.xlabel("Predicted label")
        fig.canvas.draw()

        # Convert matplotlib plot to normal image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return img
