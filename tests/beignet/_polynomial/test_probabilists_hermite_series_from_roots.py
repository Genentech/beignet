import beignet.polynomial
import torch


def test_probabilists_hermite_series_from_roots():
    res = beignet.polynomial.probabilists_hermite_series_from_roots(
        torch.tensor([]),
    )

    torch.testing.assert_close(
        beignet.polynomial.trim_probabilists_hermite_series(res, tolerance=0.000001),
        torch.tensor([1], dtype=torch.float32),
    )

    for index in range(1, 5):
        pol = beignet.polynomial.probabilists_hermite_series_from_roots(
            torch.cos(torch.linspace(-torch.pi, 0, 2 * index + 1)[1::2]),
        )

        res = beignet.polynomial.evaluate_probabilists_hermite_series_1d(
            torch.cos(torch.linspace(-torch.pi, 0, 2 * index + 1)[1::2]),
            pol,
        )

        tgt = 0

        assert len(pol) == index + 1

        torch.testing.assert_close(
            beignet.polynomial.probabilists_hermite_series_to_power_series(pol)[-1],
            torch.tensor(1, dtype=torch.float32),
        )

        torch.testing.assert_close(
            res,
            torch.tensor(tgt, dtype=torch.float32),
        )
