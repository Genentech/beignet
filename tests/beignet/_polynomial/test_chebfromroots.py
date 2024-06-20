import beignet.polynomial
import beignet.polynomial._chebfromroots
import beignet.polynomial._trim_chebyshev_series
import torch.testing


def test_chebfromroots():
    torch.testing.assert_close(
        beignet.polynomial.trim_chebyshev_series(
            beignet.polynomial._chebfromroots.chebfromroots(
                torch.tensor([]),
            ),
            tolerance=1e-6,
        ),
        torch.tensor([1.0]),
    )

    for index in range(1, 5):
        roots = torch.linspace(-torch.pi, 0, 2 * index + 1)

        roots = roots[1::2]

        roots = torch.cos(roots)

        output = beignet.polynomial._chebfromroots.chebfromroots(roots)

        torch.testing.assert_close(
            beignet.polynomial.trim_chebyshev_series(
                output * 2 ** (index - 1),
                tolerance=1e-6,
            ),
            beignet.polynomial.trim_chebyshev_series(
                torch.tensor([0] * index + [1], dtype=torch.float32),
                tolerance=1e-6,
            ),
        )
