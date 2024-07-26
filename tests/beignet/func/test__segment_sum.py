import torch
import pytest

from beignet.func._partition import _segment_sum


def test_segment_sum_basic():
    input = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    indexes = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    expected_output = torch.tensor([3, 7, 5], dtype=torch.float32)
    output = _segment_sum(input, indexes)
    assert torch.allclose(output, expected_output)


def test_segment_sum_with_n():
    input = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    indexes = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    n = 4
    expected_output = torch.tensor([3, 7, 5, 0], dtype=torch.float32)
    output = _segment_sum(input, indexes, n)
    assert torch.allclose(output, expected_output)


def test_segment_sum_multidim():
    input = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=torch.float32)
    indexes = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    expected_output = torch.tensor([[4, 6], [12, 14], [9, 10]], dtype=torch.float32)
    output = _segment_sum(input, indexes)
    assert torch.allclose(output, expected_output)


def test_segment_sum_with_kwargs():
    input = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    indexes = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    expected_output = torch.tensor([3, 7, 5], dtype=torch.float64)
    output = _segment_sum(input, indexes, dtype=torch.float64)
    assert torch.allclose(output, expected_output)


def test_segment_sum_invalid_indexes_length():
    input = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    indexes = torch.tensor([0, 0, 1], dtype=torch.int64)
    with pytest.raises(ValueError):
        _segment_sum(input, indexes)
