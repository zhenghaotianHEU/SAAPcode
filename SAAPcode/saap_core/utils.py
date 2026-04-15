import os
import random
import numpy as np
import torch
import psutil


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_memory(logger, stage):
    return


def project_root_from_file(file_path):
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(file_path))))
