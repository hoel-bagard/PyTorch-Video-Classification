import argparse
import os
import glob
import json
import shutil

import cv2
import numpy as np


def query_yes_no(question: str, default: str = "yes") -> True:
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError(f"invalid default answer: {default}")

    while True:
        print(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').")


def check_entry(entries, video_path):
    """
    Check if there is already an entry in the json file for a given video.
    Note: be careful with relative paths
    Args:
        entries: Content of the json file.
        video_path: path of the video
    Returns: True if the path is already in the file, False otherwise.
    """
    entries = np.asarray(entries)
    for entry in entries:
        if entry["file_path"] == video_path:
            return True
    return False


def make_video_timestamps(video_path):
    base_text = "Press \"d\" to toggle defect/no defect, space to go to the next frame and \"q\" to quit"
    cap = cv2.VideoCapture(video_path)
    label_time_stamps = []
    for frame_nb in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, img = cap.read()
        if ret:    
            # width, height = img.shape[0:2]
            # img = cv2.resize(img, (int(width/3), int(height/3)))
            img = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, None, 0)
            img = cv2.putText(img, base_text + f"    (defect {os.path.normpath(video_path).split(os.sep)[-2]})", (20, 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            while True:
                cv2.imshow("Frame", img)
                key = cv2.waitKey(10)
                if key == ord("q"):
                    exit()
                if key == ord("d"):
                    label_time_stamps.append(frame_nb)
                    break
                if key == 32:  # Space key
                    break
        else:
            break
    return label_time_stamps


def main():
    parser = argparse.ArgumentParser("Tool to help label videos frame by frame")
    parser.add_argument("data_path", help='Path to the dataset')
    parser.add_argument("--output_path", default=None, type=str, help='Path to where the label file will be created')
    args = parser.parse_args()

    output_path = args.output_path if args.output_path else os.path.join(args.data_path, "labels.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.isfile(output_path):
        replace = query_yes_no("Labels already exists at {output_path}, do you wish to add new entries to it ?")
        if not replace:
            print("Please give a different output path or delete existing file")
            exit()
        else:
            with open(output_path) as json_file: 
                existing_data = json.load(json_file) 
                entries = existing_data["entries"]
    else:
        # Creates dummy pre-existing data if there was none
        existing_data = {"entries": []}
        entries = existing_data["entries"]

    file_list = glob.glob(os.path.join(args.data_path, "**", "*.avi"), recursive=True)
    nb_videos: int = len(file_list)
    for i, video_path in enumerate(file_list):
        print(f"Processing video {os.path.basename(video_path)} ({i+1}/{nb_videos})", flush=True, end='\r')

        # Check if it is already in the json
        if os.path.isfile(output_path) and check_entry(entries, video_path):
            print(f"\nThere is already an entry for {video_path}, proceeding to next video")
            continue

        # Make time stamps and create json entry
        label_time_stamps = make_video_timestamps(video_path)
        json_entry = {
                        "file_path": video_path,
                        "time_stamps": label_time_stamps
                    }
        entries.append(json_entry)

    # Write everything to disk
    print(f"\nWriting labels to {output_path}")
    with open(output_path, 'w') as label_file:
        json.dump(existing_data, label_file, indent=4) 

    print("Finished labelling dataset")


if __name__ == "__main__":
    main()
