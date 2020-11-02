import torch.nn as nn
import torch.nn.functional as F

from src.networks.layers import DarknetBlock

from .network_utils import layer_init
from config.model_config import ModelConfig


class Conv3DNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = ModelConfig.CHANNELS

        self.convs = nn.Sequential(*[DarknetBlock(channels[i-1], channels[i], ModelConfig.NB_BLOCKS[i-1])
                                     for i in range(1, len(channels))])
        nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))

        self.apply(layer_init)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()

        if not ModelConfig.USE_N_TO_N:
            print("N to 1 not implemented for 3DNet")
            exit()
        else:
            print("N to n not implemented for 3DNet")
            exit()
        return F.log_softmax(x, dim=-1)
