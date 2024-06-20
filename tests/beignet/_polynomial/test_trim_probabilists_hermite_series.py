import beignet.polynomial
import pytest
import torch


def test_trim_probabilists_hermite_series():
    coef = torch.tensor([2, -1, 1, 0], dtype=torch.float32)

    with pytest.raises(ValueError):
        beignet.polynomial.trim_probabilists_hermite_series(coef, -1)

    torch.testing.assert_close(
        beignet.polynomial.trim_probabilists_hermite_series(coef), coef[:-1]
    )

    torch.testing.assert_close(
        beignet.polynomial.trim_probabilists_hermite_series(coef, 1),
        coef[:-3],
    )

    torch.testing.assert_close(
        beignet.polynomial.trim_probabilists_hermite_series(coef, 2),
        torch.tensor([0], dtype=torch.float32),
    )
