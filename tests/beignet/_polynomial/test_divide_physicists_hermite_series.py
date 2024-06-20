import beignet.polynomial
import torch


def test_divide_physicists_hermite_series():
    for i in range(5):
        for j in range(5):
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial.add_physicists_hermite_series(ci, cj)
            quo, rem = beignet.polynomial.divide_physicists_hermite_series(tgt, ci)
            res = beignet.polynomial.add_physicists_hermite_series(
                beignet.polynomial.multiply_physicists_hermite_series(quo, ci),
                rem,
            )
            torch.testing.assert_close(
                beignet.polynomial.trim_physicists_hermite_series(res, tolerance=1e-6),
                beignet.polynomial.trim_physicists_hermite_series(tgt, tolerance=1e-6),
            )
