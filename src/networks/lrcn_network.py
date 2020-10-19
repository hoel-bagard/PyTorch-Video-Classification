import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.networks.layers import (
    DarknetConv,
    DarknetBlock
)
from config.model_config import ModelConfig


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        channels = ModelConfig.CHANNELS

        self.first_conv = DarknetConv(1 if ModelConfig.USE_GRAY_SCALE else 3, ModelConfig.CHANNELS[0], 3)
        self.blocks = nn.Sequential(*[DarknetBlock(channels[i-1], channels[i], ModelConfig.NB_BLOCKS[i-1])
                                      for i in range(1, len(channels))])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        x = self.first_conv(inputs)
        for block in self.blocks:
            x = block(x)
        return x


class LRCN(nn.Module):
    def __init__(self, hidden_size: int = 60, num_layers: int = 5):
        super(LRCN, self).__init__()
        self.cnn = CNN()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.hidden_size: int = hidden_size

        self.hidden_cell = (torch.zeros(self.num_layers, ModelConfig.BATCH_SIZE, self.hidden_size, device=self.device),
                            torch.zeros(self.num_layers, ModelConfig.BATCH_SIZE, self.hidden_size, device=self.device))
        self.lstm = nn.LSTM(576, hidden_size, num_layers, batch_first=True)
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
        else:
            print("N to n not implemented for LRCN")
            exit()
        return F.log_softmax(x, dim=-1)

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
