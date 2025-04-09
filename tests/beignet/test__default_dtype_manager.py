import torch

from beignet import default_dtype_manager


def test_default_dtype_manager():
    original_dtype = torch.get_default_dtype()

    assert original_dtype != torch.float64

    with default_dtype_manager(torch.float64):
        assert torch.get_default_dtype() == torch.float64

    assert torch.get_default_dtype() == original_dtype
