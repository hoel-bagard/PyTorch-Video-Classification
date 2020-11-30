from typing import Optional

import torch

from .cnn_feature_extractor import FeatureExtractor
from .lrcn_network import LRCN
from .transformer_network import Transformer


class ModelHelper:
    LRCN = LRCN
    Transformer = Transformer


def build_model(model_type: type, model_path: Optional[str] = None, eval_mode: bool = False, **model_config):
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

    # Add a cnn feature extractor to the kwargs for the networks that need one
    model_config["feature_extractor"] = FeatureExtractor(**model_config)
    model = model_type(**model_config)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    if eval:
        model.eval()

    model = model.float()  # TODO: see if this line can be removed
    model.to(device)
    return model
