from typing import (
    Callable,
    Tuple
)

from einops import rearrange
import torch
import torch.nn as nn

from .network_utils import (
    layer_init,
    get_cnn_output_size
)
from .layers import Conv3D


class Conv3DNet(nn.Module):
    def __init__(self, feature_extractor: nn.Module, batch_size: int,
                 output_classes: int, n_to_n: bool, sequence_length: int,
                 dim_feedforward: int = 512, nlayers: int = 3,
                 layer_init: Callable[[nn.Module], None] = layer_init, **kwargs):
        """
        CNN feature extractor followed by a few 3D convolutions
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
        self.feature_extractor_output_shape: Tuple[int, int] = get_cnn_output_size(**kwargs, dense=False)

        self.feature_extractor = feature_extractor
        self.net_3D = torch.nn.Sequential(
            Conv3D(kwargs["channels"][-1], 8, kernel_size=(2, 3, 3), stride=(1, 2, 2)),
            Conv3D(8, 4, kernel_size=(2, 3, 3)),
            Conv3D(4, 4, kernel_size=(2, 3, 3)),
            Conv3D(4, output_classes, kernel_size=(2, 6, 4), activation=None, use_batch_norm=False))

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

        # if self.n_to_n:
        #     print("N to N not implemented for 3DNet")
        #     exit()
        # else:
        #     print("N to 1 not implemented for 3DNet")
        #     exit()
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
