import argparse
import json
from pathlib import Path
from random import shuffle
import shutil


def main():
    parser = argparse.ArgumentParser("Validation/Train splitting for videos who have been split into jpgs")
    parser.add_argument("data_path", type=Path, help="Path to the train dataset")
    parser.add_argument("--split", "-s", type=float, default=0.85, help="Split percentage")
    args = parser.parse_args()

    assert 0 < args.split and args.split < 1, "Split value must be between 0 and 1"

    data_path: Path = args.data_path
    val_path = data_path.parent / "Validation"
    val_entries = []
    # Instead of deleting entries in the original train labels, I just create a new json from scratch
    new_train_entries = []

    # Read train label file
    train_label_file_path = data_path / "labels.json"
    assert train_label_file_path.is_file(), "Label file is missing"
    with open(train_label_file_path) as json_file:
        train_labels = json.load(json_file)
        train_entries = train_labels["entries"]
        shuffle(train_entries)

    # Stores all the path. Moving the videos in the splitting loop is unsafe if it crashes.
    files_to_move: dict[Path, Path] = {}

    nb_labels = len(train_entries)
    for i, label in enumerate(train_entries, start=1):
        sample_base_path = data_path / label["file_path"]
        msg = f"Processing data {str(sample_base_path)}    ({i}/{nb_labels})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r")

        # Checks that the data is there, if not "remove" it from the labels
        image_paths = list(sample_base_path.glob("*.jpg"))
        # assert len(image_paths), f"Images for video {str(sample_base_path)} are missing"
        if not len(image_paths):
            continue
            # print(f"Images for video {str(sample_base_path)} are missing")

        if i >= args.split*nb_labels:
            for image_path in image_paths:
                files_to_move[image_path] = (val_path / label["file_path"]) / image_path.name
            val_entries.append(label)
        else:
            new_train_entries.append(label)

    # Move videos to Validation folder
    print("\nMoving files")
    for old_path, new_path in files_to_move.items():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(new_path.parent))

    # Delete old train labels file
    train_label_file_path.unlink()

    # Write new labels
    print("Writing train labels")
    with open(train_label_file_path, 'w') as label_file:
        json.dump({"entries": new_train_entries}, label_file, indent=4)
    print("Writing validation labels")
    with open(val_path / "labels.json", 'w') as label_file:
        json.dump({"entries": val_entries}, label_file, indent=4)

    print("Finished splitting dataset")


if __name__ == "__main__":
    main()
