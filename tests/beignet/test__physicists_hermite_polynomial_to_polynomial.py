import beignet
import torch


def test_physicists_hermite_polynomial_to_polynomial():
    coefficients = [
        torch.tensor([1.0]),
        torch.tensor([0.0, 2]),
        torch.tensor([-2.0, 0, 4]),
        torch.tensor([0.0, -12, 0, 8]),
        torch.tensor([12.0, 0, -48, 0, 16]),
        torch.tensor([0.0, 120, 0, -160, 0, 32]),
        torch.tensor([-120.0, 0, 720, 0, -480, 0, 64]),
        torch.tensor([0.0, -1680, 0, 3360, 0, -1344, 0, 128]),
        torch.tensor([1680.0, 0, -13440, 0, 13440, 0, -3584, 0, 256]),
        torch.tensor([0.0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512]),
    ]

    for index in range(10):
        torch.testing.assert_close(
            beignet.physicists_hermite_polynomial_to_polynomial(
                torch.tensor([0.0] * index + [1.0]),
            ),
            coefficients[index],
        )
