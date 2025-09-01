import pytest
import torch

import beignet.polynomials


def test_chebyshev_extrema():
    with pytest.raises(ValueError):
        beignet.polynomials.chebyshev_extrema(1.5)

    with pytest.raises(ValueError):
        beignet.polynomials.chebyshev_extrema(1)

    torch.testing.assert_close(
        beignet.polynomials.chebyshev_extrema(2),
        torch.tensor([-1.0, 1.0]),
    )

    torch.testing.assert_close(
        beignet.polynomials.chebyshev_extrema(3),
        torch.tensor([-1.0, 0.0, 1.0]),
    )

    torch.testing.assert_close(
        beignet.polynomials.chebyshev_extrema(4),
        torch.tensor([-1.0, -0.5, 0.5, 1.0]),
    )

    torch.testing.assert_close(
        beignet.polynomials.chebyshev_extrema(5),
        torch.tensor([-1.0, -0.707106781187, 0, 0.707106781187, 1.0]),
    )
