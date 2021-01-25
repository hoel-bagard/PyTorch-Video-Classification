from typing import (
    Callable,
    Union,
    Tuple
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_utils import (
    layer_init,
    get_cnn_output_size
)


class Conv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: Union[int, Tuple[int, int]] = 1, **kwargs):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm3d(out_channels, momentum=0.01)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = self.conv(input_data)
        x = self.batch_norm(x)
        x = F.leaky_relu(x, 0.1)
        return x


class Conv3DNet(nn.Module):
    def __init__(self, feature_extractor: nn.Module, batch_size: int,
                 output_classes: int, n_to_n: bool, sequence_length: int,
                 dim_feedforward: int = 512, nlayers: int = 3,
                 layer_init: Callable[[nn.Module], None] = layer_init, **kwargs):
        """
        CNN feature extractor followed by an transformer layers
        Args:
            feature_extractor: CNN to use as a feature extractor
            batch_size: batch size
            output_classes: Number of output classes (classification)
            n_to_n: Whether the model should be N_To_N or N_to_1
            sequence_length: Length of the input sequence
            dim_feedforward: Dimension of the fc layers in the transformer encoder part
            nlayers: Number of sub-encoder-layers in the encoder
            layer_init: Function used to initialise the layers of the network
        """
        super().__init__()
        self.n_to_n = n_to_n

        self.feature_extractor = feature_extractor
        # TODO: Make 3D convs that make sense
        self.conv3d = Conv3D(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))

        self.apply(layer_init)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()

        if self.n_to_n:
            print("N to N not implemented for 3DNet")
            exit()
        else:
            print("N to 1 not implemented for 3DNet")
            exit()
        # x = F.log_softmax(x, dim=-1)
        return x
