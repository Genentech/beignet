import beignet
import torch


def test_add_physicists_hermite_polynomial():
    for j in range(5):
        for k in range(5):
            target = torch.zeros(max(j, k) + 1)

            target[j] = target[j] + 1
            target[k] = target[k] + 1

            torch.testing.assert_close(
                beignet.trim_physicists_hermite_polynomial_coefficients(
                    beignet.add_physicists_hermite_polynomial(
                        torch.tensor([0.0] * j + [1.0]),
                        torch.tensor([0.0] * k + [1.0]),
                    ),
                    tol=0.000001,
                ),
                beignet.trim_physicists_hermite_polynomial_coefficients(
                    target,
                    tol=0.000001,
                ),
            )
