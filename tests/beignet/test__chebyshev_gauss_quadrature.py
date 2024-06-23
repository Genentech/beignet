import beignet.polynomial
import torch.testing


def test_chebyshev_gauss_quadrature():
    x, w = beignet.polynomial.chebyshev_gauss_quadrature(
        torch.tensor([100]),
    )

    v = beignet.polynomial.chebyshev_series_vandermonde_1d(x, 99)
    vv = torch.dot(v.T * w, v)
    vd = 1 / torch.sqrt(torch.diagonal(vv))

    torch.testing.assert_close(
        vd[:, None] * vv * vd,
        torch.eye(100),
    )

    torch.testing.assert_close(
        torch.sum(w),
        torch.pi,
    )
