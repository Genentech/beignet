import torch
from beignet.polynomial import polyval3d, polyvander3d


def test_polyvander3d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3, 4)

    output = polyvander3d(
        a,
        b,
        c,
        degree=torch.tensor([1.0, 2.0, 3.0]),
    )

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        polyval3d(
            a,
            b,
            c,
            coefficients,
        ),
    )

    output = polyvander3d(
        a,
        b,
        c,
        degree=torch.tensor([1.0, 2.0, 3.0]),
    )

    assert output.shape == (5, 24)
