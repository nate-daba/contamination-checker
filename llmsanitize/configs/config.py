import json
import torch
from pathlib import Path
from dataclasses import dataclass

from llmsanitize.utils.utils import dict_to_object
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("config")


# Read yaml config for variable configs (configuration that can change frequently)
with open(Path(__file__).parent / 'main_config.json', 'r') as rf:
    config_dict = json.load(rf)
    logger.info(config_dict)

supported_methods = {dic['name']: dic for dic in config_dict['methods']}