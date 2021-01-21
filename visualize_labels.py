import argparse
import os
from typing import (
    List,
    Tuple
)

import cv2
import numpy as np

from src.torch_utils.dataset.pytorch_video_dataset_utils import n_to_n_loader
from config.data_config import DataConfig


def show_video(video_path, labels: List[int], size: Tuple[int, int] = None) -> np.ndarray:
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

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_nb = 0
    while frame_nb < video_length:

        frame_ok, frame = cap.read()
        if not frame_ok:
            break

        # Crop
        if "video-1" in video_path:
            frame = frame[900:]

        if size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

        frame = cv2.copyMakeBorder(frame, 40, 0, 0, 0, cv2.BORDER_CONSTANT, None, 0)
        defect_text = f"The defect is: {os.path.normpath(video_path).split(os.sep)[-4]}"
        frame_text = f"    -    Frame {frame_nb} / {video_length}"
        frame = cv2.putText(frame, defect_text + frame_text, (20, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        frame = cv2.putText(frame, f"Status: defect {'visible' if labels[frame_nb] != 0 else 'non-visible'}",
                            (frame.shape[1]-300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        frame = cv2.circle(frame, (frame.shape[1]-100, 20), 15,
                           visible_color if labels[frame_nb] != 0 else non_visible_color, -1)

        while True:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(10)
            if key == 32:  # Space key, next frame
                break
            elif key == ord("q"):  # quit
                cap.release()
                cv2.destroyAllWindows()
                exit()

        frame_nb += 1

    cv2.destroyAllWindows()
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to dataset")
    parser.add_argument("--defect", aliases=['d'], default=None, type=str, help="Displays only this type of defect")
    parser.add_argument("--skip", action="store_true", help="Skips videos where the defect has been detected")
    args = parser.parse_args()

    data = n_to_n_loader(args.data_path, DataConfig.LABEL_MAP, load_videos=False)

    for sample in data:
        if args.defect and args.defect not in sample[0]:
            continue
        if args.skip and len(sample[1]) == 1:
            continue
        show_video(sample[0], sample[1])


if __name__ == "__main__":
    main()
