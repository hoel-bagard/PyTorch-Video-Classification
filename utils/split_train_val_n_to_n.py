import argparse
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser("Validation/Train splitting")
    parser.add_argument('data_path', help='Path to the train dataset')
    parser.add_argument('--split', "-s", type=float, default=0.85, help='Split percentage')
    args = parser.parse_args()

    print("The way it's implemented, splitting is done by video and not by subfolder...")
    # exit()

    assert 0 < args.split and args.split < 1, "Split value must be between 0 and 1"

    val_path = os.path.join(os.path.dirname(args.data_path.strip("/")), "Validation")
    val_labels = {"entries": []}
    val_entries = val_labels["entries"]
    # Instead of deleting entries in the original train labels, I just create a new json from scratch
    new_train_labels = {"entries": []}
    new_train_entries = new_train_labels["entries"]

    # Read train label file
    train_label_file_path = os.path.join(args.data_path, "labels.json")
    assert os.path.isfile(train_label_file_path), "Label file is missing"
    with open(train_label_file_path) as json_file:
        train_labels = json.load(json_file)
        train_entries = train_labels["entries"]

    # Stores all the path. Moving the videos in the splitting loop is unsafe if it crashes.
    videos_to_move = {}

    nb_labels = len(train_entries)
    for i, label in enumerate(train_entries, start=1):
        video_path = os.path.join(args.data_path, label["file_path"])
        msg = f"Processing data {video_path}    ({i}/{nb_labels})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r")

        assert os.path.isfile(video_path), f"Video {video_path} is missing"

        if i >= args.split*nb_labels:
            videos_to_move[video_path] = os.path.join(val_path, label["file_path"])
            val_entries.append(label)
        else:
            new_train_entries.append(label)

    # Move videos to Validation folder
    for old_path, new_path in videos_to_move.items():
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.move(old_path, os.path.dirname(new_path))

    # Delete old train labels file
    os.remove(train_label_file_path)

    # Write new labels
    print("\nWriting train labels")
    with open(train_label_file_path, 'w') as label_file:
        json.dump(new_train_labels, label_file, indent=4)
    print("Writing validation labels")
    with open(os.path.join(val_path, "labels.json"), 'w') as label_file:
        json.dump(val_labels, label_file, indent=4)

    print("Finished splitting dataset")


if __name__ == "__main__":
    main()
