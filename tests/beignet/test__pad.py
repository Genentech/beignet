import pytest
import torch

from beignet import pad_to_target_length


def test_pad_to_target_length():
    input = torch.rand(3, 4, 5)

    for dim in range(3):
        output = pad_to_target_length(input, target_length=16, dim=dim, value=0.0)
        expected_shape = (*input.shape[:dim], 16, *input.shape[dim + 1 :])

        assert output.shape == expected_shape, f"{dim=}"


def test_pad_to_target_length_raises_if_too_short():
    input = torch.rand(3, 4, 5)
    with pytest.raises(ValueError):
        pad_to_target_length(input, target_length=1, dim=0)
