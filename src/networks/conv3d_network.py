from typing import (
    Callable,
    Union,
    List,
    Tuple
)

import numpy as np
import torch
import torch.nn as nn

from src.torch_utils.networks.network_utils import (
    layer_init,
    get_cnn_output_size
)
from src.torch_utils.networks.layers import (
    Conv3D,
    Rearrange
)


class Conv3DNet(nn.Module):
    def __init__(self, feature_extractor: nn.Module, conv3d_channels: List[int],
                 conv3d_kernels: List[Union[int, Tuple[int, int, int]]], conv3d_strides: List[Union[int, Tuple[int, int, int]]],
                 conv3d_padding: List[Union[int, Tuple[int, int, int]]], output_classes: int, n_to_n: bool, sequence_length: int,
                 layer_init: Callable[[nn.Module], None] = layer_init, **kwargs):
        """
        CNN feature extractor followed by a few 3D convolutions
        Args:
            feature_extractor: CNN to use as a feature extractor
            kernels: List with the kernel size for each 3D convolution
            strides: List with the stride for each 3D convolution
            padding: List with the padding for each 3D convolution
            output_classes: Number of output classes (classification)
            n_to_n: Whether the model should be N_To_N or N_to_1
            sequence_length: Length of the input sequence
            layer_init: Function used to initialise the layers of the network
        """
        super().__init__()
        self.n_to_n = n_to_n
        self.sequence_length = sequence_length
        self.feature_extractor = feature_extractor

        # self.feature_extractor_output_shape: Tuple[int, int] = get_cnn_output_size(**kwargs, dense=False)
        conv2D_output_shape = self.feature_extractor(torch.zeros(1, 3,
                                                                 *kwargs["image_sizes"],
                                                                 device="cpu")).shape
        self.feature_extractor_output_shape: Tuple[int, int] = conv2D_output_shape[2:]

        self.input_to_conv2D = Rearrange("b t c h w -> (b t) c h w")
        self.conv2D_to_conv3D = Rearrange("b t c h w -> b c t h w")

        self.net_3D = nn.Sequential(
            *[Conv3D(conv3d_channels[i], conv3d_channels[i+1], kernel_size=conv3d_kernels[i], stride=conv3d_strides[i],
                     padding=conv3d_padding[i])
              for i in range(0, len(conv3d_channels)-1)],
            Conv3D(conv3d_channels[-1], output_classes, kernel_size=conv3d_kernels[-1], stride=conv3d_strides[-1],
                   padding=conv3d_padding[-1])
        )
        self.conv3D_to_dense = Rearrange("b c t h w -> b t h w c")

        # TODO: Adapt the 2D function to make it handle the 3D case
        conv3D_output_shape = self.net_3D(torch.zeros(1, kwargs["channels"][-1],
                                                      self.sequence_length, *self.feature_extractor_output_shape,
                                                      device="cpu")).shape

        assert not self.n_to_n or self.sequence_length == conv3D_output_shape[2], "Sequence length must not be modified"

        if self.n_to_n:
            self.dense = nn.Linear(conv3D_output_shape[1] * np.prod(conv3D_output_shape[3:]), output_classes)
        else:
            self.dense = nn.Linear(np.prod(conv3D_output_shape[1:]), output_classes)

        if layer_init:
            self.apply(layer_init)

    def forward(self, input_data):
        batch_size, timesteps, C, H, W = input_data.size()
        x = self.input_to_conv2D(input_data)
        x = self.feature_extractor(x)
        x = x.view(batch_size, timesteps, -1, *self.feature_extractor_output_shape)
        x = self.conv2D_to_conv3D(x)
        x = self.net_3D(x)
        x = self.conv3D_to_dense(x)

        x = torch.flatten(x, start_dim=2 if self.n_to_n else 1)
        x = self.dense(x)
        return x
