import os
import random

import numpy
import torch


def set_seed(seed=None):
    if seed is None:
        seed_str = os.getenv("BEIGNET_BENCHMARK_SEED", "42")
        if seed_str.startswith("0x"):
            seed = int(seed_str, 16)
        else:
            seed = int(seed_str)

    random.seed(seed)

    numpy.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False

    return seed
