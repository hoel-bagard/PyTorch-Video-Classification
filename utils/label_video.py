import argparse
import os
import pathlib
import glob
import json

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
    Note: be careful with relative paths / Windows format paths
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
    base_text = ("Press \"space\" to toggle defect/no defect, \"a\" and \"d\" to go to the previous and next frame"
                 "and \"q\" to quit")
    status = False  # False for non-visible, True for visible
    visible_color, non_visible_color = (0, 255, 0), (0, 0, 255)
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    label_time_stamps = []
    frame_nb = 0
    while frame_nb < video_length:
        ret, img = cap.read()
        if ret:
            # Crop
            img = img[900:-400]
            img = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, None, 0)
            defect_text = f"    -    The defect is: {os.path.normpath(video_path).split(os.sep)[-4]}"
            frame_text = f"    -    Frame {frame_nb} / {video_length}"
            img = cv2.putText(img, base_text + defect_text + frame_text, (20, 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            img = cv2.putText(img, f"Status: defect {'visible' if status else 'non-visible'}", (img.shape[1]-300, 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            img = cv2.circle(img, (img.shape[1]-100, 20), 15, visible_color if status else non_visible_color, -1)

            # Show image and wait for user input
            while True:
                cv2.imshow("Frame", img)
                key = cv2.waitKey(10)
                if key == 32:  # Space key, toggle
                    label_time_stamps.append(frame_nb)
                    frame_nb += 1
                    status = not status
                    break
                elif key == ord("a"):  # previous
                    if frame_nb > 0:
                        # Remove time stamps
                        if label_time_stamps != [] and label_time_stamps[-1] == frame_nb:
                            status = not status
                        label_time_stamps = [t for t in label_time_stamps if t != frame_nb]
                        # Go back a frame
                        frame_nb -= 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
                    break
                elif key == ord("d"):  # next
                    frame_nb += 1
                    break
                elif key == ord("q"):  # quit
                    cap.release()
                    cv2.destroyAllWindows()
                    return -1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return label_time_stamps


def main():
    parser = argparse.ArgumentParser("Tool to help label videos frame by frame")
    parser.add_argument("data_path", help='Path to the dataset')
    parser.add_argument("--defect", aliases=['d'], default=None, type=str,
                        help='If you wish to label one defect in particular (for exemple: "g1000")')
    parser.add_argument("--output_path", aliases=['out'], default=None, type=str,
                        help='Path to where the label file will be created')
    args = parser.parse_args()

    output_path = args.output_path if args.output_path else os.path.join(args.data_path, "labels.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.isfile(output_path):
        replace = query_yes_no(f"Labels already exists at {output_path}, do you wish to add new entries to it ?")
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

    file_list = glob.glob(os.path.join(args.data_path, "**", "*.mp4"), recursive=True)  # used to be .avi
    nb_videos: int = len(file_list)
    for i, video_path in enumerate(file_list):
        video_subpath = os.path.join(*video_path.split(os.path.sep)[-5:])  # Keeps only the "constant" part
        video_subpath = pathlib.PureWindowsPath(video_subpath).as_posix()  # Just in case the labeller is using Windows

        # When labelling only one type of defect, skip the other ones
        if args.defect and args.defect not in video_path:
            continue

        print(f"Processing video {video_path} ({i+1}/{nb_videos})", flush=True)

        # Check if it is already in the json
        if os.path.isfile(output_path) and check_entry(entries, video_subpath):
            print(f"\nThere is already an entry for {video_path}, proceeding to next video")
            continue

        while not query_yes_no("Are you ready for the next video ?", default="no"):
            continue

        # Make time stamps and create json entry
        label_time_stamps = make_video_timestamps(video_path)
        if label_time_stamps != -1:
            json_entry = {
                            "file_path": video_subpath,
                            "time_stamps": label_time_stamps
                        }
            entries.append(json_entry)
        else:
            break

    # Write everything to disk
    print(f"\nWriting labels to {output_path}")
    with open(output_path, 'w') as label_file:
        json.dump(existing_data, label_file, indent=4)

    print("Finished labelling dataset")


if __name__ == "__main__":
    main()
