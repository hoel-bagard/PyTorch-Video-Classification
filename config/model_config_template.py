class ModelConfig:
    # Training parameters
    BATCH_SIZE         = 16            # Batch size
    MAX_EPOCHS         = 2000          # Number of Epochs
    BUFFER_SIZE        = 256           # Buffer Size, used for the shuffling
    LR                 = 1e-3          # Learning Rate
    LR_DECAY           = 0.997
    DECAY_START        = 10
    REG_FACTOR         = 0.005       # Regularization factor (Used to be 0.005 for the fit mode)

    # Network part
    MODEL = "Transformer"   # Transformer or LRCN
    CHANNELS = [3, 8, 16, 32, 32, 16]
    SIZES = [3, 3, 3, 3, 3, 3]   # Kernel sizes
    STRIDES = [2, 2, 2, 2, 2, 2]
    NB_BLOCKS = [1, 2, 2, 2, 1]
    VIDEO_SIZE = 20
    USE_GRAY_SCALE = False
    IMAGE_SIZES = (256, 256)  # All images will be resized to this size
    OUTPUT_CLASSES = 101
    WORKERS = 8   # Number of workers for dataloader
