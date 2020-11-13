import os
from typing import (
    List,
    Tuple
)

import cv2
import numpy as np

from config.data_config import DataConfig
from src.dataset.dali_dataloader import DALILoader


def show_video(video, labels: List[int], size: Tuple[int, int] = None) -> np.ndarray:
    """
    Draw predictions and labels on the video to help with TensorBoard visualisation.
    Args:
        video: Raw video.
        prediction: Prediction of the network, after softmax but before taking argmax
        label: Label corresponding to the video
        size: The images will be resized to this size
    Returns: images with information written on them
    """
    visible_color, non_visible_color = (0, 255, 0), (0, 0, 255)

    for frame_nb, frame in enumerate(video):
        if size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

        frame = np.asarray(frame, dtype=np.uint8)
        frame = cv2.copyMakeBorder(frame, 40, 0, 0, 0, cv2.BORDER_CONSTANT, None, 0)
        defect_text = f"The defect is: {labels[frame_nb]}"
        frame_text = f"    -    Frame {frame_nb} / {len(video)}"
        frame = cv2.putText(frame, defect_text + frame_text, (20, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        frame = cv2.putText(frame, f"Status: defect {'visible' if labels[frame_nb] != 0 else 'non-visible'}",
                            (frame.shape[1]-300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        frame = cv2.circle(frame, (frame.shape[1]-100, 20), 15,
                           visible_color if labels[frame_nb] != 0 else non_visible_color, -1)

        cv2.imwrite(f"../temp/{frame_nb}.jpg", frame)


def main():
    train_dataloader = DALILoader(os.path.join(DataConfig.DATA_PATH, "Train"), DataConfig.LABEL_MAP, limit=10)

    for i, batch in enumerate(train_dataloader):
        batch = batch[0]  # For some reason there's a dimension there....
        data = batch["video"].cpu()
        label_batch = batch["label"].cpu()
        for j, (video, label) in enumerate(zip(data, label_batch)):
            if label != 0:
                label = label.repeat(len(video))  # DALI cannot have sequences with mixed labels =(
                print(f"Shape of video {j} from batch {i}: {video.shape}")
                show_video(video, label)
                exit()


if __name__ == '__main__':
    main()
