import os

from config.data_config import DataConfig
from src.dataset.dali_test import dali_n_to_n


def main():
    label_list = dali_n_to_n(os.path.join(DataConfig.DATA_PATH, "Train"), DataConfig.LABEL_MAP, limit=10)
    print(label_list)


if __name__ == '__main__':
    main()
