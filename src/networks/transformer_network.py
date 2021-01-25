from typing import Callable

from einops import rearrange
import torch.nn as nn

from .transformer_layer import TransformerLayer
from .network_utils import (
    layer_init,
    get_cnn_output_size
)


class Transformer(nn.Module):
    def __init__(self, feature_extractor: nn.Module, batch_size: int,
                 output_classes: int, n_to_n: bool, sequence_length: int,
                 dim_feedforward: int = 512, n_layers: int = 3,
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
        self.feature_extractor = feature_extractor
        self.n_to_n = n_to_n
        self.cnn_output_size = get_cnn_output_size(**kwargs, output_channels=kwargs["channels"][-1])

        self.transformer = TransformerLayer(self.cnn_output_size, output_classes,
                                            dim_feedforward=dim_feedforward, nlayers=n_layers)
        self.dense = nn.Linear(sequence_length * output_classes, output_classes)

    def forward(self, inputs):
        batch_size, timesteps, C, H, W = inputs.size()
        x = rearrange(inputs, "b t  c h w -> (b t) c h w")
        x = self.feature_extractor(x)
        x = x.view(batch_size, timesteps, -1)
        x = self.transformer(x)   # Outputs (batch_size, timesteps, nb_classes)
        if not self.n_to_n:
            x = x.view(batch_size, -1)
            x = self.dense(x)  # Outputs (batch_size, nb_classes)
        # x = F.log_softmax(x, dim=-1)
        return x
