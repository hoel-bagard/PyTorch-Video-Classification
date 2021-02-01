import argparse
import os
import glob
import shutil


def main():
    parser = argparse.ArgumentParser("Validation/Train splitting")
    parser.add_argument('data_path', help='Path to the train dataset')
    args = parser.parse_args()

    val_path = os.path.join(os.path.dirname(args.data_path.strip("/")), "Validation")
    os.makedirs(val_path, exist_ok=True)

    # Build a map between id and names
    label_map = {}
    with open(os.path.join(args.data_path, "..", "classes.names")) as table_file:
        for key, line in enumerate(table_file):
            label = line.strip()
            label_map[key] = label

    for key in range(len(label_map)):
        os.makedirs(os.path.join(val_path, label_map[key]))
        file_list = glob.glob(os.path.join(args.data_path, label_map[key], "*.avi"))
        nb_files = len(file_list)
        for i, file_path in enumerate(file_list):
            print(f"Processing image {os.path.basename(file_path)} ({i+1}/{nb_files})", end='\r')
            if i >= 0.85*nb_files:
                shutil.move(file_path, os.path.join(val_path, label_map[key], os.path.basename(file_path)))
        print(f"Finished splitting {label_map[key]}"
              + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns-len("Finished splitting {label_map[key]}")))
    print("\nFinished splitting dataset")


if __name__ == "__main__":
    main()
