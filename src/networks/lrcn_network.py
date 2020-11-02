import torch
import torch.nn as nn

from src.networks.layers import DarknetConv
from .network_utils import (
    layer_init,
    get_cnn_output_size
)
from config.model_config import ModelConfig


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = ModelConfig.OUTPUT_CLASSES
        channels = ModelConfig.CHANNELS
        sizes = ModelConfig.SIZES
        strides = ModelConfig.STRIDES

        self.first_conv = DarknetConv(1 if ModelConfig.USE_GRAY_SCALE else 3, channels[0], sizes[0], stride=strides[0],
                                      padding=ModelConfig.PADDINGS[0])
        self.blocks = nn.Sequential(*[DarknetConv(channels[i-1], channels[i], sizes[i], stride=strides[i],
                                                  padding=ModelConfig.PADDINGS[i])
                                    for i in range(1, len(channels))])

        self.apply(layer_init)

    def forward(self, inputs):
        x = self.first_conv(inputs)
        for block in self.blocks:
            x = block(x)
        x = torch.flatten(x, start_dim=1)
        return x


class LRCN(nn.Module):
    def __init__(self, hidden_size: int = 60, num_layers: int = 5):
        super().__init__()
        self.cnn = CNN()
        cnn_output_size = get_cnn_output_size()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.hidden_size: int = hidden_size

        self.hidden_cell = (torch.zeros(self.num_layers, ModelConfig.BATCH_SIZE, self.hidden_size, device=self.device),
                            torch.zeros(self.num_layers, ModelConfig.BATCH_SIZE, self.hidden_size, device=self.device))
        self.lstm = nn.LSTM(cnn_output_size, self.hidden_size, num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, ModelConfig.OUTPUT_CLASSES)

    def forward(self, inputs):
        batch_size, timesteps, C, H, W = inputs.size()
        x = inputs.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, timesteps, -1)
        x, self.hidden_cell = self.lstm(x, self.hidden_cell)
        if not ModelConfig.USE_N_TO_N:
            x = x[:, -1]
        x = self.dense(x)
        # F.log_softmax(x, dim=-1)
        return x

    def reset_lstm_state(self, batch_size: int = ModelConfig.BATCH_SIZE):
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
