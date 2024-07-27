import torch

from beignet.func._interact import _safe_sum


def test_safe_sum_complex():
    x = torch.tensor([1 + 2j, 3 + 4j, 5 + 6j], dtype=torch.complex64)

    result = _safe_sum(x)

    expected = torch.tensor(9 + 12j, dtype=torch.complex64)

    assert torch.allclose(result, expected)


def test_safe_sum_floating_point():
    x = torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32)

    result = _safe_sum(x)

    expected = torch.tensor(6.6, dtype=torch.float32)

    assert torch.allclose(result, expected)


def test_safe_sum_integer():
    x = torch.tensor([1, 2, 3], dtype=torch.int32)

    result = _safe_sum(x)

    expected = torch.tensor(6, dtype=torch.int32)

    assert torch.equal(result, expected)


def test_safe_sum_with_dim():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

    result = _safe_sum(x, dimension=0)

    expected = torch.tensor([4, 6], dtype=torch.float32)

    assert torch.allclose(result, expected)


def test_safe_sum_with_keepdim():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

    result = _safe_sum(x, dimension=0, keep_dimension=True)

    expected = torch.tensor([[4, 6]], dtype=torch.float32)

    assert torch.allclose(result, expected)
