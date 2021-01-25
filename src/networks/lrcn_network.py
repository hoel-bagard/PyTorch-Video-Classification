from typing import Callable

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.Functional as F

from config.model_config import ModelConfig
from src.torch_utils.networks.network_utils import (
    layer_init,
    get_cnn_output_size
)
from src.torch_utils.utils.tensorboard import TensorBoard


class LRCN(nn.Module):
    def __init__(self, feature_extractor: nn.Module, batch_size: int,
                 output_classes: int, n_to_n: bool,
                 hidden_size: int = 60, num_layers: int = 5,
                 layer_init: Callable[[nn.Module], None] = layer_init, **kwargs):
        """
        CNN feature extractor followed by an LSTM
        Args:
            feature_extractor: CNN to use as a feature extractor
            batch_size: batch size
            output_classes: Number of output classes (classification)
            n_to_n: Whether the model should be N_To_N or N_to_1
            layer_init: Function used to initialise the layers of the network
        """
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        cnn_output_size = get_cnn_output_size(**kwargs, output_channels=kwargs["channels"][-1])

        self.feature_extractor = feature_extractor
        self.n_to_n = n_to_n
        self.num_layers = num_layers
        self.hidden_size: int = hidden_size

        self.hidden_cell = (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
                            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))
        self.lstm = nn.LSTM(cnn_output_size, self.hidden_size, num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_classes)

    def forward(self, inputs):
        batch_size, timesteps, C, H, W = inputs.size()
        x = rearrange(inputs, "b t c h w -> (b t) c h w")
        x = self.feature_extractor(x)
        x = x.view(batch_size, timesteps, -1)
        x, self.hidden_cell = self.lstm(x, self.hidden_cell)
        if not self.n_to_n:
            x = x[:, -1]
        x = self.dense(x)
        # F.log_softmax(x, dim=-1)
        return x

    def reset_lstm_state(self, batch_size: int):
        """
        Args:
            batch_size: Needs to be the same as the size of the next batch.
                        Can be smaller if the last batch does not have enough elements.
        """
        # Debug. Slows down training.
        # del self.hidden_cell
        # torch.cuda.empty_cache()

        self.hidden_cell = (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
                            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))

    @staticmethod
    def preprocess(tensorboard: TensorBoard, videos, labels) -> (torch.Tensor, torch.Tensor):
        # LSTM needs proper batches (the pytorch implementation at least)
        batch_size = videos.size()[0]
        videos = F.pad(videos, (0, 0, 0, 0, 0, 0, 0, ModelConfig.BATCH_SIZE-batch_size))
        tensorboard.model.reset_lstm_state(videos.shape[0])
        return videos, labels
