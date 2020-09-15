import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.networks.layers import (
    DarknetConv,
    DarknetBlock
)
from src.networks.transformer_layer import TransformerLayer
from config.model_config import ModelConfig


def get_cnn_output_size():
    width, height = ModelConfig.IMAGE_SIZES
    for kernel_size, stride in zip(ModelConfig.SIZES, ModelConfig.STRIDES):
        width = ((width - kernel_size) // stride) + 1

    for kernel_size, stride in zip(ModelConfig.SIZES, ModelConfig.STRIDES):
        height = ((height - kernel_size) // stride) + 1

    return width*height*ModelConfig.CHANNELS[-1]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.output_size = ModelConfig.OUTPUT_CLASSES
        channels = ModelConfig.CHANNELS
        sizes = ModelConfig.SIZES
        strides = ModelConfig.STRIDES

        self.first_conv = DarknetConv(1 if ModelConfig.USE_GRAY_SCALE else 3, channels[0], sizes[0], stride=strides[0])
        self.blocks = nn.Sequential(*[DarknetConv(channels[i-1], channels[i], sizes[i], stride=strides[i])
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
        x = torch.flatten(x, start_dim=1)
        return x


class Transformer(nn.Module):
    def __init__(self, nb_classes: int = 2, hidden_size: int = 60, num_layers: int = 5):
        super(Transformer, self).__init__()
        self.cnn_output_size = get_cnn_output_size()
        self.cnn = CNN()
        self.transformer = TransformerLayer(self.cnn_output_size, nb_classes)
        self.dense = nn.Linear(ModelConfig.VIDEO_SIZE * nb_classes, nb_classes)

    def forward(self, inputs):
        batch_size, timesteps, C, H, W = inputs.size()
        x = inputs.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, timesteps, -1)
        x = self.transformer(x)   # Outputs (batch_size, timesteps, nb_classes)
        x = x.view(batch_size, -1)
        x = self.dense(x)  # Outputs (batch_size, nb_classes)
        x = F.log_softmax(x, dim=-1)
        return x
