import os


class DataConfig:
    # Recording part
    DATA_PATH          = 'path/to/dataset'  # Path to the dataset folder
    USE_CHECKPOINT     = True               # Whether to save checkpoints or not
    CHECKPOINT_DIR     = 'path/to/checkpoint_dir/AI_Name'  # Path to checkpoint dir
    CHECKPT_SAVE_FREQ  = 10                  # How often to save checkpoints (if they are better than the previous one)
    KEEP_CHECKPOINTS   = False                # Whether to remove the checkpoint dir
    USE_TB             = True                # Whether generate a TensorBoard or not
    TB_DIR             = 'path/to/log_dir/AI_Name'  # TensorBoard dir
    KEEP_TB            = False                # Whether to remove the TensorBoard dir
    VAL_FREQ           = 5                  # How often to compute accuracy and images (also used for validation freq)
    RECORD_START       = 0                  # Checkpoints and TensorBoard are not recorded before this epoch

    # Dataloading part
    DALI = False  # Whether to use DALI for data loading or PyTorch/numpy
    NUM_WORKERS = 12   # Number of workers for PyTorch dataloader

    # Build a map between id and names
    LABEL_MAP = {}
    with open(os.path.join(DATA_PATH, "classes.names")) as table_file:
        for key, line in enumerate(table_file):
            label = line.strip()
            LABEL_MAP[key] = label
    OUTPUT_CLASSES = len(LABEL_MAP)
