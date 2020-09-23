import argparse
import os
import glob
import json
import shutil

import cv2


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



def main():
    parser = argparse.ArgumentParser("Tool to help label videos frame by frame")
    parser.add_argument("data_path", help='Path to the dataset')
    parser.add_argument("--output", default=None, type=str, help='Output folder path')
    args = parser.parse_args()

    output_path = args.data_path if args.output else os.path.join(args.data_path, "labels")
    os.makedirs(output_path, exist_ok=True)

    file_list = glob.glob(os.path.join(args.data_path, "**", "*.avi"), recursive=True)
    nb_videos = len(file_list)
    base_text = "Press \"d\" to toggle defect/no defect, space to go to the next frame and \"q\" to quit"

    for i, video_path in enumerate(file_list):
        print(f"Processing image {os.path.basename(video_path)} ({i+1}/{nb_videos})", flush=True, end='\r')

        label_path = os.path.join(output_path, os.path.join(*video_path.split(os.sep)[-3:]) + ".json")
        if os.path.isfile(label_path):
            replace = query_yes_no("Label for this video already exists, do you wish to replace it ?")
            if not replace:
                continue

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

        with open(label_path, 'w') as label_file:
            json_text = json.dumps({
                                     "file_path": video_path,
                                     "time_stamps": label_time_stamps,
                                    }, indent=4)
            label_file.write(json_text)

    print("\nFinished labelling dataset")


if __name__ == "__main__":
    main()
