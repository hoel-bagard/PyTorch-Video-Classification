import torch
import torch.nn as nn

from src.networks.layers import (
    DarknetConv,
)
from src.networks.transformer_layer import TransformerLayer
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


class Transformer(nn.Module):
    def __init__(self, hidden_size: int = 60, num_layers: int = 5):
        super().__init__()
        nb_classes = ModelConfig.OUTPUT_CLASSES
        self.cnn_output_size = get_cnn_output_size()
        self.cnn = CNN()
        self.transformer = TransformerLayer(self.cnn_output_size, nb_classes, dim_feedforward=512, nlayers=2)
        self.dense = nn.Linear(ModelConfig.VIDEO_SIZE * nb_classes, nb_classes)

    def forward(self, inputs):
        batch_size, timesteps, C, H, W = inputs.size()
        x = inputs.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, timesteps, -1)
        x = self.transformer(x)   # Outputs (batch_size, timesteps, nb_classes)
        if not ModelConfig.USE_N_TO_N:
            x = x.view(batch_size, -1)
            x = self.dense(x)  # Outputs (batch_size, nb_classes)
        # x = F.log_softmax(x, dim=-1)
        return x
