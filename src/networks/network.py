import math

import torch
import torch.nn as nn

from src.networks.layers import (
    DarknetConv,
    DarknetBlock
)
from config.model_config import ModelConfig


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.output_size = 2
        channels = ModelConfig.CHANNELS

        self.first_conv = DarknetConv(3, ModelConfig.CHANNELS[0], 3)
        self.blocks = nn.Sequential(*[DarknetBlock(channels[i-1], channels[i], ModelConfig.NB_BLOCKS[i-1])
                                      for i in range(1, len(channels))])
        self.last_conv = nn.Conv2d(ModelConfig.CHANNELS[-1], self.output_size, 6, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):

        # Use conv3D with 1 for the kernel depth ?
        inputs = torch.transpose(inputs, 0, 1)  # Swaps seq_len and batch
        for seq_element in inputs:
            x = self.first_conv(inputs)
            for block in self.blocks:
                x = block(x)
            x = self.last_conv(x)
            x = torch.flatten(x, start_dim=1)

        # LSTM here

        return x
