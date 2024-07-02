import torch
from beignet.polynomial import polyval2d, polyvander2d


def test_polyvander2d():
    a, b, c = torch.rand(3, 5) * 2 - 1

    coefficients = torch.rand(2, 3)

    output = polyvander2d(a, b, degree=torch.tensor([1, 2]))

    torch.testing.assert_close(
        output @ torch.ravel(coefficients),
        polyval2d(
            a,
            b,
            coefficients,
        ),
    )

    output = polyvander2d(
        a,
        b,
        degree=torch.tensor([1, 2]),
    )

    assert output.shape == (5, 6)
