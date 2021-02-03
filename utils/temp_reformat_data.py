import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()

    data_path: Path = args.data_path
    image_paths = list(data_path.rglob("*.jpg"))

    nb_imgs = len(image_paths)
    for i, image_path in enumerate(image_paths, start=1):
        msg = f"Processing data {str(image_path)}    ({i}/{nb_imgs})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r")

        # Puts videos' images in a folder for each video
        # dest_path = image_path.parent / str(image_path.name)[:7] / str(image_path.name[8:])
        # dest_path.parent.mkdir(parents=True, exist_ok=True)
        # shutil.move(image_path, dest_path)

        # Rename images
        # dest_path = image_path.parent / (str(image_path.name.stem) + str(image_path.name.suffix).zfill(3))
        # dest_path.parent.mkdir(parents=True, exist_ok=True)
        # shutil.move(image_path, dest_path)

    print("\nFinished renaming images")


if __name__ == "__main__":
    main()
