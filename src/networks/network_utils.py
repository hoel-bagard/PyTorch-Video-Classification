from typing import (
    Tuple,
    List
)
import math

import torch
import torch.nn as nn


def layer_init(layer, weight_gain: float = 1, bias_const: float = 0,
               weights_init: str = 'xavier', bias_init: str = 'zeros'):
    """
    Layer initialisation function.
    Most of it comes from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py.
    Args:
        layer: layer to be initialized.
        weight_gain:
        bias_const:
        weights_init: Can be 'xavier', "orthogonal" or 'uniform'.
        bias_init: Can be 'zeros', 'uniform'.
    """

    if isinstance(layer, nn.Linear):
        if weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        if bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)
    if isinstance(layer, nn.Conv2d):
        n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        layer.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()


def get_cnn_output_size(image_sizes: Tuple[int, int], sizes: List[int], strides: List[int], paddings: List[int],
                        output_channels: int, **kwargs) -> int:
    """
    Computes the output size of a cnn  (flattened)
    Args:
        image_sizes: Dimensions of the input image (width, height).
        sizes: List with the kernel size for each convolution
        strides: List with the stride for each convolution
        padding: List with the padding for each convolution
        output_channels: Number of output channels of the last convolution
    """
    width, height = image_sizes
    for kernel_size, stride, padding in zip(sizes, strides, paddings):
        width = ((width - kernel_size + 2*padding) // stride) + 1

    for kernel_size, stride, padding in zip(sizes, strides, paddings):
        height = ((height - kernel_size + 2*padding) // stride) + 1

    return width*height*output_channels
