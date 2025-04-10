import math

import pytest
import torch

import beignet
from beignet import default_dtype_manager


@pytest.mark.parametrize("dtype", [torch.float64])
def test_gauss_physicists_hermite_polynomial_quadrature(dtype):
    with default_dtype_manager(dtype):
        x, w = beignet.gauss_physicists_hermite_polynomial_quadrature(100)

        v = beignet.physicists_hermite_polynomial_vandermonde(x, 99)
        vv = (v.T * w) @ v
        vd = 1 / torch.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        torch.testing.assert_close(
            vv,
            torch.eye(100),
        )

        torch.testing.assert_close(
            w.sum(),
            torch.tensor(math.sqrt(math.pi)),
        )
