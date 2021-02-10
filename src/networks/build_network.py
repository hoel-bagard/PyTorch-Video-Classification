from typing import Optional

import torch

from src.torch_utils.networks.cnn_feature_extractor import (
    CNNFeatureExtractor,
    DarknetFeatureExtrator
)
from .lrcn_network import LRCN
from .transformer_network import Transformer
from .conv3d_network import Conv3DNet
from .cnn_wrapper import CNN_Wrapper


class ModelHelper:
    LRCN = LRCN
    Transformer = Transformer
    Conv3DNet = Conv3DNet
    CNN_Wrapper = CNN_Wrapper


class FeatureExtractorHelper:
    SimpleCNN = CNNFeatureExtractor
    DarknetCNN = DarknetFeatureExtrator


def build_model(model_type: type, output_classes: bool, model_path: Optional[str] = None,
                eval_mode: bool = False, **kwargs):
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

    kwargs["output_classes"] = output_classes
    # Instanciate a cnn feature extractor to the kwargs for the networks that need one
    if kwargs["feature_extractor"]:
        kwargs["feature_extractor"] = kwargs["feature_extractor"](**kwargs)
    model = model_type(**kwargs)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    if eval:
        model.eval()

    model.to(device)
    return model
