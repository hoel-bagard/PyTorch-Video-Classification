from typing import (
    Callable,
    Tuple
)

import numpy as np
import torch
import torch.nn as nn

from src.torch_utils.networks.network_utils import layer_init
from src.torch_utils.networks.layers import Rearrange


class CNN_Wrapper(nn.Module):
    def __init__(self, feature_extractor: nn.Module, output_classes: int, sequence_length: int,
                 layer_init: Callable[[nn.Module], None] = layer_init, **kwargs):
        """
        Wrapper to use when setting sequence length to 1 (i.e. when doing image classification)
        CNN feature extractor followed by a few 3D convolutions
        Args:
            feature_extractor: CNN to use as a feature extractor
            output_classes: Number of output classes (classification)
            sequence_lenght: Length of the input sequence
            layer_init: Function used to initialise the layers of the network
        """
        super().__init__()
        assert sequence_length == 1, "Sequence length must be one to use the feature extractor wrapper"

        self.input_to_conv2D = Rearrange("b t c h w -> (b t) c h w")
        self.feature_extractor = feature_extractor
        conv2D_output_shape = self.feature_extractor(torch.zeros(1, 3,
                                                                 *kwargs["image_sizes"],
                                                                 device="cpu")).shape
        self.feature_extractor_output_shape: Tuple[int, int] = conv2D_output_shape[2:]
        self.dense = nn.Linear(np.prod(conv2D_output_shape[1:]), output_classes)

        if layer_init:
            self.apply(layer_init)

    def forward(self, input_data):
        batch_size, timesteps, C, H, W = input_data.size()
        x = self.input_to_conv2D(input_data)
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        return x
