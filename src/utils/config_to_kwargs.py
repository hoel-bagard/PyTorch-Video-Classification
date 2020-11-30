from typing import (
    Dict,
    Any
)

from config.data_config import DataConfig
from config.model_config import ModelConfig


def get_data_config_dict() -> Dict:
    # return dict(filter(lambda attr: not attr[0].startswith('__') and attr[0][0].isupper(), vars(DataConfig).items()))
    return dict([(key.lower(), value) for key, value in vars(DataConfig).items()
                 if not key.startswith('__') and key[0].isupper()])


def get_model_config_dict() -> Dict[str, Any]:
    config_attribute_dict = vars(ModelConfig)

    model_config_dict = {}
    for key, value in config_attribute_dict.items():
        if not key.startswith('__') and key[0].isupper():
            model_config_dict[key.lower()] = value

    return model_config_dict
