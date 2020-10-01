import os
import glob
from typing import Dict

import numpy as np
import cv2

from config.model_config import ModelConfig


def default_loader(data_path: str, label_map: Dict, limit: int = None, load_videos: bool = False) -> np.ndarray:
    """
    Args:
        data_path: Path to the root folder of the dataset.
                   This folder is expected to contain subfolders for each class, with the videos inside.
        label_map: dictionarry mapping an int to a class
        limit (int, optional): If given then the number of elements for each class in the dataset
                            will be capped to this number
        load_videos: If true then this function returns the videos instead of their paths
    Return:
        numpy array containing the paths/videos and the associated label
    """
    data = []
    for key in range(len(label_map)):
        file_types = ("*.avi", "*.mp4")
        pathname = os.path.join(data_path, label_map[key], "**")
        video_paths = []
        [video_paths.extend(glob.glob(os.path.join(pathname, ext), recursive=True)) for ext in file_types]
        for i, video_path in enumerate(video_paths):
            msg = f"Loading data {video_path}    ({i}/{len(video_paths)}) for class {label_map[key]}"
            print(msg + ' ' * (os.get_terminal_size()[0] - len(msg)), end="\r")
            if load_videos:
                cap = cv2.VideoCapture(video_path)
                video = []
                while(cap.isOpened()):
                    frame_ok, frame = cap.read()
                    if frame_ok:
                        if ModelConfig.USE_GRAY_SCALE:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = np.expand_dims(frame, -1)  # To keep a channel dimension (gray scale)
                        else:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video.append(frame)
                    else:
                        break
                cap.release()
                data.append([np.asarray(video), key])
            else:
                data.append([video_path, key])
            if limit and i >= limit:
                break

    data = np.asarray(data)
    return data
