import torch

import beignet


def test_add_laguerre_polynomial():
    for i in range(5):
        for j in range(5):
            target = torch.zeros(max(i, j) + 1)

            target[i] = target[i] + 1
            target[j] = target[j] + 1

            torch.testing.assert_close(
                beignet.trim_laguerre_polynomial_coefficients(
                    beignet.add_laguerre_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                        torch.tensor([0.0] * j + [1.0]),
                    ),
                    tol=0.000001,
                ),
                beignet.trim_laguerre_polynomial_coefficients(
                    target,
                    tol=0.000001,
                ),
            )
