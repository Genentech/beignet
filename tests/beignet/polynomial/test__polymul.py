import torch
from beignet.polynomial import polymul, polytrim


def test_polymul():
    for j in range(5):
        for k in range(5):
            target = torch.zeros(j + k + 1)

            target[j + k] = target[j + k] + 1

            torch.testing.assert_close(
                polytrim(
                    polymul(
                        torch.tensor([0.0] * j + [1.0]),
                        torch.tensor([0.0] * k + [1.0]),
                    ),
                    tol=0.000001,
                ),
                polytrim(
                    target,
                    tol=0.000001,
                ),
            )
