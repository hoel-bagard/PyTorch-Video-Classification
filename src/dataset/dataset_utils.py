import os
import glob
import json
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


def read_label(label: str, label_map: Dict, video_length: int):
    """
    Args:
        label: Entry from the label file, must contain path and time stamps
        label_map: dictionarry mapping an int to a class
        video_length: length of the video
    Return:
        array with the class for each frame
    """
    for key in range((len(label_map))):
        if label_map[key] in label["file_path"]:
            video_cls = key
            break
    assert video_cls, f"There is no class corresponding to {label['file_path']}"

    for key, value in label_map.items():
        if(value == "not_visible"):
            not_visible_cls = key
            break
    assert not_visible_cls, "There should be a 'not_visible' class"

    visibility_status = False
    label["time_stamps"].append(-1)
    labels = np.full(video_length, not_visible_cls)
    for i in range(len(label["time_stamps"])-1):
        visibility_status = not visibility_status
        if visibility_status:
            labels[label["time_stamps"][i]:label["time_stamps"][i+1]] = video_cls

    return np.asarray(labels)


def n_to_n_loader(data_path: str, label_map: Dict, limit: int = None, load_videos: bool = False) -> np.ndarray:
    """
    Loading function for when every frame has an associated label
    Args:
        data_path: Path to the root folder of the dataset.
                   This folder is expected to contain subfolders for each class, with the videos inside.
                   It should also contain a label.json file with the labels (file paths and time stamps)
        label_map: dictionarry mapping an int to a class
        limit (int, optional): If given then the number of elements for each class in the dataset
                            will be capped to this number
        load_videos: If true then this function returns the videos instead of their paths
    Return:
        numpy array containing the paths/videos and the associated labels
    """

    # Read label file
    label_file_path = os.path.join(data_path, "labels.json")
    assert os.path.isfile(label_file_path), "Label file is missing"

    with open(label_file_path) as json_file:
        json_labels = json.load(json_file)
        labels = json_labels["entries"]

    nb_labels = len(labels)
    data = []
    for i, label in enumerate(labels):
        video_path = os.path.join(data_path, label["file_path"])
        msg = f"Loading data {video_path}    ({i}/{len(nb_labels)})"
        print(msg + ' ' * (os.get_terminal_size()[0] - len(msg)), end="\r")

        assert os.path.isfile(video_path), f"Video {video_path} is missing"
        cap = cv2.VideoCapture(video_path)
        video_length: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        label = read_label(label, label_map, video_length)

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
            data.append([np.asarray(video), label])
        else:
            data.append([video_path, label])
        if limit and i >= limit:
            break

    data = np.asarray(data)
    return data
