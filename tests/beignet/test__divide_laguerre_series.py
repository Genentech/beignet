import beignet.polynomial
import torch


def test_divide_laguerre_series():
    for i in range(5):
        for j in range(5):
            quo, rem = beignet.polynomial.divide_laguerre_series(
                beignet.polynomial.add_laguerre_series(
                    torch.tensor([0] * i + [1]),
                    torch.tensor([0] * j + [1]),
                ),
                torch.tensor([0] * i + [1]),
            )

            torch.testing.assert_close(
                beignet.polynomial.trim_laguerre_series(
                    beignet.polynomial.add_laguerre_series(
                        beignet.polynomial.multiply_laguerre_series(
                            quo,
                            torch.tensor([0] * i + [1]),
                        ),
                        rem,
                    ),
                    tolerance=0.000001,
                ),
                beignet.polynomial.trim_laguerre_series(
                    beignet.polynomial.add_laguerre_series(
                        torch.tensor([0] * i + [1]),
                        torch.tensor([0] * j + [1]),
                    ),
                    tolerance=0.000001,
                ),
            )
