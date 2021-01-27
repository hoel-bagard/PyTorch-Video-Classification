from typing import (
    Union,
    Tuple
)

from src.torch_utils.networks.build_network import (
    ModelHelper,
    FeatureExtractorHelper
)


class ModelConfig:
    # Training parameters
    BATCH_SIZE         = 4              # Batch size
    MAX_EPOCHS         = 2000           # Number of Epochs
    BUFFER_SIZE        = 10*BATCH_SIZE  # Buffer Size, used for the shuffling
    LR                 = 4e-3           # Learning Rate
    LR_DECAY           = 0.997
    DECAY_START        = 10
    REG_FACTOR         = 0.005          # Used for weight regularisation (L2 penalty)

    # Data processing
    SEQUENCE_LENGTH  = 15          # Video size / Number of frames in each sample
    IMAGE_SIZES      = (256, 256)  # All images will be resized to this size
    GRAYSCALE        = False       # Cannot be used if using the DALI dataloader, converts images to grayscale

    # Video specific
    N_TO_N = True        # If True then there must be a label for each frame

    # Network part
    MODEL           = ModelHelper.Transformer

    # Feature Extractor
    FEATURE_EXTRACTOR = FeatureExtractorHelper.SimpleCNN  # Model to use to extract features from images
    # First value should be the number of channels in the input image,
    # there should be one more value in CHANNELS than in the other lists
    CHANNELS = [1 if GRAYSCALE else 3, 24, 32, 64, 64, 48, 32, 24, 16, 12]
    SIZES = [3, 3, 3, 3, 3, 3, 3, 3, 3]   # Kernel sizes
    STRIDES = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    PADDINGS = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    NB_BLOCKS = [2, 2, 2, 1, 1, 1, 1, 1, 1]  # Used if feature extractor is the darknet one

    # Network specific
    CONV3D_CHANNELS: list[int] = [CHANNELS[-1], 8, 4, 4]  # Number of input channels for each 3Dconv
    CONV3D_KERNELS: list[Union[int, Tuple[int, int, int]]] = [(2, 3, 3), (2, 3, 3), (2, 3, 3), (2, 6, 4)]
    CONV3D_STRIDES: list[Union[int, Tuple[int, int, int]]]  = [(1, 2, 2), 1, 1, 1]
