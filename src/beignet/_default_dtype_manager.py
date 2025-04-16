from contextlib import contextmanager

import torch


@contextmanager
def default_dtype_manager(dtype):
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(original_dtype)
