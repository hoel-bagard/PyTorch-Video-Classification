from typing import (
    Tuple,
    Dict,
    Optional
)

import cv2
from einops import rearrange
import numpy as np
import torch


def draw_pred(videos: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor,
              label_map: Dict[int, str], n_to_n: bool,
              size: Optional[Tuple[int, int]] = None, ) -> np.ndarray:
    """
    Draw predictions and labels on the image to help with TensorBoard visualisation.
    Args:
        videos: Raw videos.
        predictions: Predictions of the network, after softmax but before taking argmax
        labels: Labels corresponding to the images
        label_map: Dictionary linking class index to class name
        n_to_n: True if using one label for each element of the sequence
        size: If given, the images will be resized to this size
    Returns: images with information written on them
    """
    videos: np.ndarray = videos.cpu().detach().numpy()
    labels: np.ndarray = labels.cpu().detach().numpy()
    predictions: np.ndarray = predictions.cpu().detach().numpy()

    frame_to_keep = len(labels) // 2

    # TODO: do that before calling the function, that way it'll be a generate draw_pred_img func
    imgs = videos[:, frame_to_keep, :, :, :]

    imgs = rearrange(imgs, 'b c w h -> b w h c')  # imgs.transpose(0, 2, 3, 1)
    if n_to_n:
        predictions = predictions[:, frame_to_keep]
        labels = labels[:, frame_to_keep]

    new_imgs = []
    for img, preds, label in zip(imgs, predictions, labels):
        img = np.asarray(img * 255.0, dtype=np.uint8)

        # TODO: Might not work, there used to always be a resize (even if to same size)
        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        # Just print the top 3 classes
        # Gets indices of top 3 pred
        nb_to_keep = 3 if len(preds) > 3 else 2
        idx = np.argpartition(preds, -nb_to_keep)[-nb_to_keep:]
        idx = idx[np.argsort(preds[idx])][::-1]
        preds = str([label_map[i] + f":  {round(float(preds[i]), 2)}" for i in idx])

        img = cv2.putText(img, preds, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, f"Label: {label}  ({label_map[label]})", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        new_imgs.append(img)
    return np.asarray(new_imgs)


def draw_pred_video(video: torch.Tensor, prediction: torch.Tensor, label: torch.Tensor,
                    label_map: Dict[int, str], n_to_n: bool = False,
                    size: Optional[Tuple[int, int]] = None, ) -> np.ndarray:
    """
    Draw predictions and labels on the video to help with TensorBoard visualisation.
    Args:
        video: Raw video.
        prediction: Prediction of the network, after softmax but before taking argmax
        label: Label corresponding to the video
        label_map: Dictionary linking class index to class name
        n_to_n: True if using one label for each element of the sequence
        size: If given, the images will be resized to this size
    Returns: images with information written on them
    """
    video: np.ndarray = video.cpu().detach().numpy()
    labels: np.ndarray = label.cpu().detach().numpy()
    preds: np.ndarray = prediction.cpu().detach().numpy()
    if not n_to_n:
        labels = np.broadcast_to(labels, video.shape[0])
        preds = np.broadcast_to(preds, (video.shape[0], preds.shape[0]))

    video = video.transpose(0, 2, 3, 1)  # Conversion to H x W x C

    new_video = []
    for img, preds, label in zip(video, preds, labels):
        img = np.asarray(img * 255.0, dtype=np.uint8)

        # TODO: Might not work, there used to always be a resize (even if to same size)
        if size:
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

    # Keep a channel dimension if in gray scale mode
    if new_video.ndim == 3:
        new_video = np.expand_dims(new_video, -1)

    return new_video
