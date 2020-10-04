from typing import Tuple

import cv2
import numpy as np
import torch

from config.data_config import DataConfig
from config.model_config import ModelConfig


def draw_pred(videos: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor,
              size: Tuple[int, int] = None) -> np.ndarray:
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

    label_map = DataConfig.LABEL_MAP
    # TODO: remove this temp fix and do it properly
    if not size:
        size = ModelConfig.IMAGE_SIZES

    new_imgs = []
    for img, preds, label in zip(imgs, predictions, labels):
        img = np.asarray(img * 255.0, dtype=np.uint8)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        # If there are too many classes, just print the top 3 ones
        if len(preds) > 5:
            # Gets indices of top 3 pred
            idx = np.argpartition(preds, -3)[-3:]
            idx = idx[np.argsort(preds[idx])][::-1]
            preds = str([label_map[i] + f":  {round(float(preds[i]), 2)}" for i in idx])
        else:
            preds = str([round(float(conf), 2) for conf in preds]) + f"  ==> {np.argmax(preds)}"

        img = cv2.putText(img, preds, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, f"Label: {label}  ({label_map[label]})", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        new_imgs.append(img)
    return np.asarray(new_imgs)


def draw_pred_video(video: torch.Tensor, prediction: torch.Tensor, label: torch.Tensor,
                    size: Tuple[int, int] = None) -> np.ndarray:
    """
    Draw predictions and labels on the video to help with TensorBoard visualisation.
    Args:
        video: Raw video.
        prediction: Prediction of the network, after softmax but before taking argmax
        label: Label corresponding to the video
        size: The images will be resized to this size
    Returns: images with information written on them
    """
    video: np.ndarray = video.cpu().detach().numpy()
    label: int = int(label.cpu().detach().numpy())
    preds: np.ndarray = prediction.cpu().detach().numpy()

    video = video.transpose(0, 2, 3, 1)  # Conversion to H x W x C

    label_map = DataConfig.LABEL_MAP
    # TODO: remove this temp fix and do it properly
    if not size:
        size = ModelConfig.IMAGE_SIZES

    new_video = []
    for img in video:
        img = np.asarray(img * 255.0, dtype=np.uint8)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        # If there are too many classes, just print the top 3 ones
        if len(preds) > 5:
            # Gets indices of top 3 pred
            idx = np.argpartition(preds, -3)[-3:]
            idx = idx[np.argsort(preds[idx])][::-1]
            preds_text = str([label_map[i] + f":  {round(float(preds[i]), 2)}" for i in idx])
        else:
            preds_text = str([round(float(conf), 2) for conf in preds]) + f"  ==> {np.argmax(preds)}"

        img = cv2.putText(img, preds_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, f"Label: {label}  ({label_map[label]})", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        new_video.append(img)

    new_video = np.asarray(new_video)
    if ModelConfig.USE_GRAY_SCALE:
        new_video = np.expand_dims(new_video, -1)  # To keep a channel dimension (gray scale)

    return new_video
