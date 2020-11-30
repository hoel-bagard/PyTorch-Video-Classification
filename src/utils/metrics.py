import itertools
from typing import (
    List,
    Dict
)


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.dataset.build_dataloader import Dataloader


class Metrics:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, train_dataloader: Dataloader, val_dataloader: Dataloader,
                 label_map: Dict[int, str], n_to_n: bool, output_classes: int, max_batches: int = 10):
        """
        Class computing usefull metrics for classification tasks
        Args:
            model: The PyTorch model being trained
            loss_fn: Function used to compute the loss of the model
            train_dataloader: DataLoader with a PyTorch DataLoader like interface, contains train data
            val_dataloader: DataLoader with a PyTorch DataLoader like interface, contains validation data
            label_map: Dictionary linking class index to class name
            n_to_n: True if using one label for each element of the sequence
            output_classes: Number of classes the network is classifying
            max_batches: If not None, then the metrics will be computed using at most this number of batches
        """
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.label_map = label_map
        self.n_to_n = n_to_n
        self.output_classes = output_classes
        self.max_batches = max_batches

    def compute_confusion_matrix(self, mode: str = "Train"):
        """
        Computes the confusion matrix. This function has to be called before using the get functions.
        Args:
            mode: Either "Train" or "Validation"
        """
        self.cm = np.zeros((self.output_classes, self.output_classes))
        for step, batch in enumerate(self.train_dataloader if mode == "Train" else self.val_dataloader, start=1):
            imgs, labels_batch = batch["video"].float(), batch["label"].cpu().detach().numpy()
            predictions_batch = self.model(imgs.to(self.device))
            predictions_batch = torch.nn.functional.softmax(predictions_batch, dim=-1)
            predictions_batch = torch.argmax(predictions_batch, dim=-1).int().cpu().detach().numpy()

            for (label_video, pred_video) in zip(labels_batch, predictions_batch):  # batch
                if self.n_to_n:
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
        per_class_acc = [self.cm[i, i] / max(1, np.sum(self.cm[i])) for i in range(len(self.cm))]
        return per_class_acc

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Returns an image containing the plotted confusion matrix.
        Taken from: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12

        Returns:
            per_class_acc: Image of the confusion matrix.
        """
        cm = self.cm
        class_names = self.label_map.values()

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
