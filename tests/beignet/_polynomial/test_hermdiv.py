import beignet.polynomial
import beignet.polynomial._add_physicists_hermite_series
import beignet.polynomial._divide_physicists_hermite_series
import beignet.polynomial._multiply_physicists_hermite_series
import beignet.polynomial._trim_physicists_hermite_series
import torch


def test_hermdiv():
    for i in range(5):
        for j in range(5):
            msg = f"At i={i}, j={j}"
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = beignet.polynomial._hermadd.add_physicists_hermite_series(ci, cj)
            quo, rem = beignet.polynomial._hermdiv.divide_physicists_hermite_series(
                tgt, ci
            )
            res = beignet.polynomial._hermadd.add_physicists_hermite_series(
                beignet.polynomial._hermmul.multiply_physicists_hermite_series(quo, ci),
                rem,
            )
            torch.testing.assert_close(
                beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                    res, tolerance=1e-6
                ),
                beignet.polynomial._hermtrim.trim_physicists_hermite_series(
                    tgt, tolerance=1e-6
                ),
                err_msg=msg,
            )
