from pathlib import Path


class DataConfig:
    # Recording part
    DATA_PATH: Path      = Path("path", "to", "dataset")  # Path to the dataset folder
    USE_CHECKPOINT       = True               # Whether to save checkpoints or not
    CHECKPOINT_DIR: Path = Path("path", "to", "checkpoint_dir", "AI_Name")  # Path to checkpoint dir
    CHECKPT_SAVE_FREQ    = 10                  # How often to save checkpoints (when the loss went down)
    KEEP_CHECKPOINTS     = False                # Whether to remove the checkpoint dir
    USE_TB               = True                # Whether generate a TensorBoard or not
    TB_DIR: Path         = Path("path", "to", "log_dir", "AI_Name")  # TensorBoard dir
    KEEP_TB              = False                # Whether to remove the TensorBoard dir
    VAL_FREQ             = 5                  # How often to compute accuracy and images (also used for validation freq)
    RECORD_START         = 0                  # Checkpoints and TensorBoard are not recorded before this epoch

    # Dataloading part
    DALI = False  # Whether to use DALI for data loading or PyTorch/numpy
    DALI_DEVICE_ID = 1  # If using DALI, which GPU to run it on (shards not suported yet)
    LOAD_FROM_IMAGES = True  # If the videos have been cut into images
    NUM_WORKERS = 12   # Number of workers for PyTorch dataloader

    # Classification part
    # Build a map between id and names
    LABEL_MAP = {}
    with open(DATA_PATH / "classes.names") as table_file:
        for key, line in enumerate(table_file):
            label = line.strip()
            LABEL_MAP[key] = label
    OUTPUT_CLASSES = len(LABEL_MAP)
