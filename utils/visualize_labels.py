import argparse
import os
import shutil
from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Final
)
import json

import cv2
import numpy as np


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
        visibility_status = not visibility_status
        if visibility_status:
            if i != len(label["time_stamps"])-1:
                labels[label["time_stamps"][i]:label["time_stamps"][i+1]] = video_cls
            else:
                labels[label["time_stamps"][i]:] = video_cls
    return np.asarray(labels)


def show_video(video_path, labels: List[int], size: Optional[Tuple[int, int]] = None) -> None:
    """
    Draw predictions and labels on the video to help with TensorBoard visualisation.
    Args:
        video: Raw video.
        prediction: Prediction of the network, after softmax but before taking argmax
        label: Label corresponding to the video
        size: The images will be resized to this size
    Returns: images with information written on them
    """
    visible_color: Final[Tuple[int, int, int]] = (0, 255, 0)
    non_visible_color: Final[Tuple[int, int, int]] = (0, 0, 255)

    cap = cv2.VideoCapture(video_path)
    video_length: Final[int] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_nb = 0
    while frame_nb < video_length:

        frame_ok, frame = cap.read()
        if not frame_ok:
            break

        # Crop, uncomment and change the values if needed.
        # if "video-1" in video_path:
        #     frame = frame[900:]

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
    parser.add_argument("--defect", '--d', default=None, type=str, help="Displays only this type of defect")
    args = parser.parse_args()

    label_map = {}
    with open(os.path.join(args.data_path, "classes.names")) as table_file:
        for key, line in enumerate(table_file):
            label = line.strip()
            label_map[key] = label

    # Read label file
    label_file_path = os.path.join(args.data_path, "labels.json")
    assert os.path.isfile(label_file_path), "Label file is missing"

    with open(label_file_path) as json_file:
        json_labels = json.load(json_file)
        labels = json_labels["entries"]

    nb_labels = len(labels)
    for i, label in enumerate(labels, start=1):
        video_path = os.path.join(args.data_path, label["file_path"])
        if args.defect and args.defect not in video_path:
            continue

        assert os.path.isfile(video_path), f"Video {video_path} is missing"
        msg = f"Loading data {video_path}    ({i}/{nb_labels})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r")

        # Get number of frame in the video
        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        label = read_n_to_n_label(label, label_map, video_length)
        show_video(video_path, label)


if __name__ == "__main__":
    main()
