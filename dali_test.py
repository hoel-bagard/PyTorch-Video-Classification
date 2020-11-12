import os

from config.data_config import DataConfig
from src.dataset.dali_test import DALILoader


def main():
    train_dataloader = DALILoader(os.path.join(DataConfig.DATA_PATH, "Train"), DataConfig.LABEL_MAP, limit=10)

    for i, batch in enumerate(train_dataloader):
        print(batch.shape)


if __name__ == '__main__':
    main()
