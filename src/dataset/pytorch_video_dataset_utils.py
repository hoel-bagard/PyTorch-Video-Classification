import json
from pathlib import Path
from os.path import sep
from typing import (
    Dict,
    Optional,
    Tuple
)

import numpy as np
import cv2

from src.torch_utils.utils.misc import clean_print


def n_to_1_loader(data_path: Path, label_map: Dict[int, str], limit: Optional[int] = None,
                  load_videos: bool = False, grayscale: bool = True) -> np.ndarray:
    """
    Args:
        data_path: Path to the root folder of the dataset.
                   This folder is expected to contain subfolders for each class, with the videos inside.
        label_map: dictionarry mapping an int to a class
        limit (int, optional): If given then the number of elements for each class in the dataset
                            will be capped to this number
        load_videos: If true then this function returns the videos instead of their paths
        grayscale: If set to true and using the load_videos option, images will be converted to grayscale
    Return:
        numpy array containing the paths/videos and the associated label
    """
    data = []
    for key in range(len(label_map)):
        file_types = (Path("*.avi"), Path("*.mp4"))
        pathname = data_path / label_map[key]
        video_paths = []
        [video_paths.extend(list(pathname.glob("**" / ext))) for ext in file_types]
        for i, video_path in enumerate(video_paths):
            clean_print(f"Loading data {str(video_path)}    ({i}/{len(video_paths)}) for class {label_map[key]}")
            if load_videos:
                cap = cv2.VideoCapture(video_path)
                video = []
                while(cap.isOpened()):
                    frame_ok, frame = cap.read()
                    if frame_ok:
                        if grayscale:
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
            if limit and len(data) == limit:
                return np.asarray(data)

    return np.asarray(data)


def read_n_to_n_label(label: str, label_map: Dict[int, str], video_length: int) -> np.ndarray:
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
    assert 'video_cls' in locals(), f"There is no class corresponding to {label['file_path']}"

    for key, value in label_map.items():
        if(value == "not_visible"):
            not_visible_cls = key
            break
    assert "not_visible_cls" in locals(), "There should be a 'not_visible' class"

    visibility_status = False
    labels = np.full(video_length, not_visible_cls)
    for i in range(len(label["time_stamps"])):
        # Skip duplicates if there happens to be some
        if i > 0 and label["time_stamps"][i] == label["time_stamps"][i-1]:
            continue
        visibility_status = not visibility_status
        if visibility_status:
            if i != len(label["time_stamps"])-1:
                labels[label["time_stamps"][i]:label["time_stamps"][i+1]] = video_cls
            else:
                labels[label["time_stamps"][i]:] = video_cls
    return np.asarray(labels, dtype=np.uint8)


def n_to_n_loader(data_path: Path, label_map: Dict[int, str], limit: Optional[int] = None,
                  load_videos: bool = False, grayscale: bool = False) -> np.ndarray:
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
        grayscale: If set to true and using the load_videos option, images will be converted to grayscale
    Return:
        numpy array containing the paths/videos and the associated labels
    """
    # Read label file
    label_file_path = data_path / "labels.json"
    assert label_file_path.is_file(), "Label file is missing"

    with open(label_file_path) as json_file:
        json_labels = json.load(json_file)
        labels = json_labels["entries"]

    nb_labels = len(labels)
    data = []
    for i, label in enumerate(labels, start=1):
        video_path = data_path / label["file_path"]
        clean_print(f"Loading data {str(video_path)}    ({i}/{nb_labels})")

        assert video_path.is_file(), f"Video {video_path} is missing"
        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        label = read_n_to_n_label(label, label_map, video_length)

        if load_videos:
            cap = cv2.VideoCapture(str(video_path))
            video = []
            while(cap.isOpened()):
                frame_ok, frame = cap.read()
                if frame_ok:
                    if grayscale:
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
        if limit and i == limit:
            break

    data = np.asarray(data, dtype=object)
    return data


def n_to_n_loader_from_images(data_path: Path, label_map: Dict[int, str], sequence_length: int,
                              limit: Optional[int] = None, load_videos: bool = False,
                              filters: Optional[list[str]] = None, grayscale: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loading function for when every frame has an associated label
    Args:
        data_path: Path to the root folder of the dataset.
                   This folder is expected to contain subfolders for each class, with the videos inside.
                   It should also contain a label.json file with the labels (file paths and time stamps)
        label_map: dictionarry mapping an int to a class
        sequence_length: Length of the sequences fed to the network
        limit (int, optional): If given then the number of elements for each class in the dataset
                            will be capped to this number
        load_videos: If true then this function returns the videos instead of their paths
        filters: Filters data whose path include the given filters (for exemple: ["subfolder1", "class2"])
        grayscale: If set to true and using the load_videos option, images will be converted to grayscale
    Return:
        numpy array containing the paths/videos and the associated labels
    """
    # Read label file
    label_file_path = data_path / "labels.json"
    assert label_file_path.is_file(), "Label file is missing"

    with open(label_file_path) as json_file:
        json_labels = json.load(json_file)
        labels = json_labels["entries"]

    nb_labels = len(labels)
    dataset_data = []
    dataset_labels = []
    for i, label in enumerate(labels, start=1):
        if filters and not any(f in label["file_path"].split(sep) for f in filters):
            continue

        sample_base_path = data_path / label["file_path"]
        clean_print(f"Loading data {str(sample_base_path)}    ({i}/{nb_labels})", end="\r")

        image_paths = list(sample_base_path.glob("*.jpg"))
        image_paths = sorted([str(image_path) for image_path in image_paths])

        assert len(image_paths), f"Images for video {str(sample_base_path)} are missing"

        label = read_n_to_n_label(label, label_map, len(image_paths))

        for start_index in range(0, len(label)-sequence_length):
            if load_videos:
                dataset_data.append([cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                                       for image_path in image_paths[start_index:start_index+sequence_length]])
            else:
                dataset_data.append(image_paths[start_index:start_index+sequence_length])
            dataset_labels.append(label[start_index:start_index+sequence_length])

            if limit and len(dataset_labels) >= limit:
                break
        if limit and len(dataset_labels) >= limit:
            break


    return dataset_data, dataset_labels
