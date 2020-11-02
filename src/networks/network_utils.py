import math

import torch
import torch.nn as nn

from .lrcn_network import LRCN
from .transformer_network import Transformer


def build_model(name: str, model_path: str = None, eval: bool = False):
    """
    Creates model corresponding to the given name.
    Args:
        name: Name of the model to create, must be one of the implemented models
        model_path: If given, then the weights will be load that checkpoint
        eval: Whether the model will be used for evaluation or not
    Returns:
        model: PyTorch model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert name in ("LRCN", "Transformer")
    if name == "LRCN":
        model = LRCN()
    elif name == "Transformer":
        model = Transformer()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    if eval:
        model.eval()

    model = model.float()
    model.to(device)
    return model


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
