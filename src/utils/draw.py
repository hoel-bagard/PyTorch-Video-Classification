import os
from typing import Tuple

import cv2
import numpy as np
import torch

from config.data_config import DataConfig


def draw_pred(videos: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor,
              size: Tuple[int, int] = (480, 480), data_path: str = DataConfig.DATA_PATH) -> np.ndarray:
    """
    Draw predictions and labels on the image to help with TensorBoard visualisation.
    Args:
        videos: Raw videos.
        predictions: Predictions of the network, after softmax but before taking argmax
        labels: Labels corresponding to the images
        size: The images will be resized to this size
    Returns: images with information written on them
    """
    videos: np.ndarray = videos.cpu().detach().numpy()
    labels: np.ndarray = labels.cpu().detach().numpy()
    predictions: np.ndarray = predictions.cpu().detach().numpy()

    imgs = videos[:, 0, :, :, :]
    imgs = imgs.transpose(0, 2, 3, 1)  # Conversion to H x W x C

    label_map = {}
    with open(os.path.join(data_path, "classes.names")) as table_file:
        for key, line in enumerate(table_file):
            label = line.strip()
            label_map[key] = label

    new_imgs = []
    for img, preds, label in zip(imgs, predictions, labels):
        img = np.asarray(img * 255.0, dtype=np.uint8)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        preds = str([round(float(conf), 2) for conf in preds]) + f"  ==> {np.argmax(preds)}"

        img = cv2.putText(img, preds, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, f"Label: {label}  ({label_map[label]})", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        new_imgs.append(img)
    return np.asarray(new_imgs)
