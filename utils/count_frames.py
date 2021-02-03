import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser("Displays stats on the dataset")
    parser.add_argument("data_path", type=Path, help="Path to the root of the data folder")
    args = parser.parse_args()

    nb_frames = []
    for video_path in list(args.data_path.glob("**/*.avi")):
        cap = cv2.VideoCapture(str(video_path))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        nb_frames.append(frame_count)
        # print(f"Frames: {frame_count}")

    print(f"Min: {np.amin(nb_frames)}")
    print(f"Max: {np.amax(nb_frames)}")
    print(f"Mean: {np.mean(nb_frames)}")
    print(f"Std: {np.std(nb_frames)}")


if __name__ == "__main__":
    main()
