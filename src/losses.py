import torch
import torch.nn as nn

from config.data_config import DataConfig


class CE_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        labels_one_hot = torch.nn.functional.one_hot(labels, DataConfig.OUTPUT_CLASSES)
        loss = torch.mean(- labels_one_hot * torch.nn.functional.log_softmax(pred, -1))

        return loss
