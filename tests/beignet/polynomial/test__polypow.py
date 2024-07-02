import functools

import torch
from beignet.polynomial import polymul, polypow, polytrim


def test_polypow():
    for i in range(5):
        for j in range(5):
            torch.testing.assert_close(
                polytrim(
                    polypow(
                        torch.arange(0.0, i + 1),
                        j,
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    functools.reduce(
                        polymul,
                        [torch.arange(0.0, i + 1)] * j,
                        torch.tensor([1.0]),
                    ),
                    tol=0.000001,
                ),
            )
