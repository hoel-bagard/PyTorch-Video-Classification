import argparse
from pathlib import Path
import json
import shutil


def main():
    parser = argparse.ArgumentParser("Validation/Train splitting")
    parser.add_argument("data_path", type=Path, help="Path to the data directory")
    parser.add_argument("label_path", type=Path, help="Path to the json file with the labels")
    parser.add_argument("output_path", type=Path, help="Path to where the non-labelled files will be moved")
    args = parser.parse_args()

    with open(args.output_path) as json_file:
        existing_data = json.load(json_file)
        labelled_video_paths = [entry["file_path"] for entry in existing_data["entries"]]

    data_path: Path = Path(args.data_path)
    output_path: Path = Path(args.output_path)
    video_paths = list(data_path.rglob("*.avi"))
    nb_videos = len(video_paths)

    for i, video_path in enumerate(video_paths, start=1):
        msg = f"Processing file {video_path},   ({i}/{nb_videos})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r")

        video_name = video_path.relative_to(data_path)
        if video_name not in labelled_video_paths:
            shutil.move(video_path, output_path / video_name)

    print("\nFinished processing dataset")


if __name__ == "__main__":
    main()
