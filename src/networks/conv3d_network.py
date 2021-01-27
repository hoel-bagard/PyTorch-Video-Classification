from typing import (
    Callable,
    Tuple
)

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn

from src.torch_utils.networks.network_utils import (
    layer_init,
    get_cnn_output_size
)
from config.model_config import ModelConfig
from src.torch_utils.networks.layers import Conv3D


class Conv3DNet(nn.Module):
    def __init__(self, feature_extractor: nn.Module, output_classes: int, n_to_n: bool, sequence_length: int,
                 layer_init: Callable[[nn.Module], None] = layer_init, **kwargs):
        """
        CNN feature extractor followed by a few 3D convolutions
        Args:
            feature_extractor: CNN to use as a feature extractor
            output_classes: Number of output classes (classification)
            n_to_n: Whether the model should be N_To_N or N_to_1
            sequence_length: Length of the input sequence
            layer_init: Function used to initialise the layers of the network
        """
        super().__init__()
        self.n_to_n = n_to_n
        self.sequence_length = sequence_length
        self.feature_extractor_output_shape: Tuple[int, int] = get_cnn_output_size(**kwargs, dense=False)

        self.feature_extractor = feature_extractor

        conv3D_channels = ModelConfig.CONV3D_CHANNELS
        conv3D_kernels = ModelConfig.CONV3D_KERNELS
        conv3D_strides = ModelConfig.CONV3D_STRIDES
        # TODO: Add padding support
        self.blocks = nn.Sequential(
            *[Conv3D(conv3D_channels[i], conv3D_channels[i+1], kernel_size=conv3D_kernels[i], stride=conv3D_strides[i])
              for i in range(0, len(conv3D_channels)-1)],
            # Default activation and BN since there is a dense layer at the end
            Conv3D(conv3D_channels[-1], output_classes, kernel_size=conv3D_kernels[-1], stride=conv3D_strides[-1])
        )

        conv3D_output_shape = self.net_3D(torch.zeros(1, ModelConfig.CHANNELS[-1],
                                                      ModelConfig.SEQUENCE_LENGTH, *self.feature_extractor_output_shape,
                                                      device="cpu")).shape
        assert not self.n_to_n or self.sequence_length == conv3D_output_shape[2], "Sequence length must not be modified"

        if self.n_to_n:
            self.dense = nn.Linear(np.prod(conv3D_output_shape[1:]), output_classes)
        else:
            self.dense = nn.Linear(conv3D_output_shape[1] * np.prod(conv3D_output_shape[:3]), output_classes)

        if layer_init:
            self.apply(layer_init)

    def forward(self, input_data):
        batch_size, timesteps, C, H, W = input_data.size()
        x = rearrange(input_data, "b t c h w -> (b t) c h w")
        x = self.feature_extractor(x)
        x = x.view(batch_size, timesteps, -1, *self.feature_extractor_output_shape)
        x = rearrange(x, "b t c h w -> b c t h w")
        print(f"Shape before 3D conv {x.size()}")
        x = self.net_3D(x)
        x = rearrange(x, "b c t h w -> b t h w c")

        if self.n_to_n:
            x = torch.flatten(x, start_dim=2)
            x = self.dense(x)
        else:
            x = torch.flatten(x, start_dim=1)
            x = self.dense(x)
        return x





# class Network3d(nn.Module):

#     def __init__(self, input_sample: torch.Tensor, class_count: int, pre_process=None):
#         super().__init__()
#         self.class_count = class_count
#         self.pre_process = pre_process

#         self.input_shape = input_sample.shape[-3:]
#         self.sequence_len = input_sample.shape[1]
#         self.feature_extractor = torch.nn.Sequential(
#             Conv2d(3, 32, kernel_size=5, stride=4),
#             Conv2d(32, 32, kernel_size=5, stride=(2, 4)),
#             Conv2d(32, 16, stride=(2, 3)),
#             Conv2d(16, 8))

#         self.feature_shape = self.feature_extractor(input_sample.view(-1, *self.input_shape).float().cpu()).shape[1:]
#         self.detector = torch.nn.Sequential(
#             Conv3d(8, 8, kernel_size=(2, 3, 3), stride=(1, 2, 2)),
#             Conv3d(8, 4, kernel_size=(2, 3, 3)),
#             Conv3d(4, 4, kernel_size=(2, 3, 3)),
#             Conv3d(4, class_count, kernel_size=(2, 6, 4), activation=None, batch_norm=False))

#     def forward(self, input_data: torch.Tensor) -> torch.Tensor:
#         out = self.feature_extractor(input_data.view(-1, *self.input_shape))
#         # print(out.shape)
#         # out = out.view(-1, self.sequence_len, *self.feature_shape).permute(0, 2, 1, 3, 4)
#         # for layer in self.detector:
#         #     out = layer(out)
#         #     print(out.shape)
#         out = self.detector(out.view(-1, self.sequence_len, *self.feature_shape).permute(0, 2, 1, 3, 4))
#         return out.view(-1, self.class_count)
