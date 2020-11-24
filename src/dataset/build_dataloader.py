import src.dataset.pytorch_transforms as transforms
from config.data_config import DataConfig
from config.model_config import ModelConfig


class Dataloader:
    def __init__(self, data_path: str, transform=None, limit: int = None, load_videos: bool = True):
        mode = "Train" if "Train" in data_path else "Validation"

        if DataConfig.DALI:
            dataloader = DALILoader(data_path,
                                    DataConfig.LABEL_MAP,
                                    limit=args.limit,
                                    mode=mode)

        else:
            if mode == "Train":
                data_transforms = Compose([
                    transforms.RandomCrop(),
                    transforms.Resize(*ModelConfig.IMAGE_SIZES),
                    transforms.Normalize(),
                    transforms.VerticalFlip(),
                    transforms.HorizontalFlip(),
                    transforms.Rotate180(),
                    transforms.ReverseTime(),
                    transforms.ToTensor(),
                    transforms.Noise()
                ])
            else:
                data_transforms=Compose([
                    transforms.Resize(*ModelConfig.IMAGE_SIZES),
                    transforms.Normalize(),
                    transforms.ToTensor()
                ])

            dataset = Dataset(data_path,
                              limit=limit,
                              load_videos=load_videos,
                              transform=data_transforms)
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=ModelConfig.BATCH_SIZE,
                                                     shuffle=(mode == "Train"),
                                                     num_workers=ModelConfig.WORKERS,
                                                     drop_last=(ModelConfig.NETWORK == "LRCN"))

        msg = f"{mode} data loaded"
        print(msg + ' ' * (os.get_terminal_size()[0] - len(msg)))

