import beignet
import torch


def test_subtract_probabilists_hermite_polynomial():
    for j in range(5):
        for k in range(5):
            target = torch.zeros(max(j, k) + 1)

            target[j] = target[j] + 1
            target[k] = target[k] - 1

            input = torch.tensor([0.0] * j + [1.0])
            other = torch.tensor([0.0] * k + [1.0])

            torch.testing.assert_close(
                beignet.trim_probabilists_hermite_polynomial_coefficients(
                    beignet.subtract_probabilists_hermite_polynomial(
                        input,
                        other,
                    ),
                    tol=0.000001,
                ),
                beignet.trim_probabilists_hermite_polynomial_coefficients(
                    target,
                    tol=0.000001,
                ),
            )
