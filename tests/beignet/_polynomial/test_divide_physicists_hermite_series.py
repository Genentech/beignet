import beignet.polynomial
import torch


def test_divide_physicists_hermite_series():
    for j in range(5):
        for k in range(5):
            quo, rem = beignet.polynomial.divide_physicists_hermite_series(
                beignet.polynomial.add_physicists_hermite_series(
                    torch.tensor([0] * j + [1]),
                    torch.tensor([0] * k + [1]),
                ),
                torch.tensor([0] * j + [1]),
            )

            torch.testing.assert_close(
                beignet.polynomial.trim_physicists_hermite_series(
                    beignet.polynomial.add_physicists_hermite_series(
                        beignet.polynomial.multiply_physicists_hermite_series(
                            quo,
                            torch.tensor([0] * j + [1]),
                        ),
                        rem,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_physicists_hermite_series(
                    beignet.polynomial.add_physicists_hermite_series(
                        torch.tensor([0] * j + [1]),
                        torch.tensor([0] * k + [1]),
                    ),
                    tolerance=1e-6,
                ),
            )
