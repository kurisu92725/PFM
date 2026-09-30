import logging
import os
import random
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict


def load_config(path):
    with open(path, 'r') as handle:
        return EasyDict(yaml.safe_load(handle))


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    name = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix:
        name = f'{prefix}_{name}'
    if tag:
        name = f'{name}_{tag}'
    path = os.path.join(root, name)
    os.makedirs(path)
    return path


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
