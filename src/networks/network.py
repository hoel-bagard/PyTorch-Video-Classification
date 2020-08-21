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

        self.first_conv = DarknetConv(1, ModelConfig.CHANNELS[0], 3)
        self.blocks = nn.Sequential(*[DarknetBlock(channels[i-1], channels[i], ModelConfig.NB_BLOCKS[i-1])
                                      for i in range(1, len(channels))])
        # self.last_conv = nn.Conv2d(ModelConfig.CHANNELS[-1], self.output_size, 6, bias=False)

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
        # x = self.last_conv(x)
        return x


class Network(nn.Module):
    def __init__(self, nb_classes: int = 2):
        super(Network, self).__init__()
        self.output_size = nb_classes
        self.cnn = CNN()
        self.lstm = nn.LSTM(288, self.output_size, 5, batch_first=True)

    def forward(self, inputs):
        batch_size, timesteps, C, H, W = inputs.size()
        x = inputs.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, timesteps, -1)
        x, _ = self.lstm(x)
        return F.log_softmax(x, dim=1)
