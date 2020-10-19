import argparse

from src.dataset.dataset_utils import n_to_n_loader
from config.data_config import DataConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to dataset")
    args = parser.parse_args()

    data = n_to_n_loader(args.data_path, DataConfig.LABEL_MAP, limit=2, load_videos=False)


if __name__ == "__main__":
    main()
