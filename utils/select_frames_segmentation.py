from argparse import ArgumentParser
from pathlib import Path
import json
import shutil

import cv2
import numpy as np


def read_n_to_n_label(label: str, label_map: dict[int, str], video_length: int) -> np.ndarray:
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
    return np.asarray(labels)


def main():
    parser = ArgumentParser()
    parser.add_argument("data_path", type=Path, help="Path to dataset")
    parser.add_argument("output_path", type=Path, help="Path to the output folder")
    parser.add_argument("--defect", '--d', default=None, type=str, help="Use only this type of defect")
    parser.add_argument("--nb_samples", '--s', default=10, type=int, help="Number of samples to use")
    parser.add_argument("--nb_frames", '--f', default=20, type=int, help="Number of frames to take from each sample")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = args.output_path

    label_map = {}
    with open(data_path / "classes.names") as table_file:
        for key, line in enumerate(table_file):
            label = line.strip()
            label_map[key] = label

    # Read label file
    label_file_path = data_path / "labels.json"
    assert label_file_path.is_file(), "Label file is missing"
    with open(label_file_path) as json_file:
        json_labels = json.load(json_file)
        labels = json_labels["entries"]

    nb_labels = len(labels)
    for i, label in enumerate(labels, start=1):
        video_path = data_path / label["file_path"]
        if args.defect and args.defect not in video_path:
            continue

        assert video_path.is_file(), f"Video {video_path} is missing"
        msg = f"Loading data {video_path}    ({i}/{nb_labels})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r")

        # Get number of frame in the video
        cap = cv2.VideoCapture(str(video_path))
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        label = read_n_to_n_label(label, label_map, video_length)
        nb_frames_to_skip = np.count_nonzero(label) // args.nb_frames  # Assumes not_visible is 0
        non_zero_frames_seen = 0
        for frame_nb in range(video_length):
            frame_ok, frame = cap.read()
            if not frame_ok:
                break

            if label[frame_nb]:  # Assumes not_visible is 0
                if non_zero_frames_seen % nb_frames_to_skip == 0:
                    frame_output_path = ((output_path / label["file_path"])
                                         .with_stem(f"{video_path.stem}_{frame_nb}")
                                         .with_suffix(".jpg"))
                    cv2.imwrite(str(frame_output_path), frame)
                non_zero_frames_seen += 1
        cap.release()
        if i == args.nb_samples:
            break


if __name__ == "__main__":
    main()
