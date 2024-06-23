import beignet.polynomial
import torch.testing

polynomial_coefficients = [
    torch.tensor([1], dtype=torch.float32),
    torch.tensor([0, 1], dtype=torch.float32),
    torch.tensor([-1, 0, 2], dtype=torch.float32),
    torch.tensor([0, -3, 0, 4], dtype=torch.float32),
    torch.tensor([1, 0, -8, 0, 8], dtype=torch.float32),
    torch.tensor([0, 5, 0, -20, 0, 16], dtype=torch.float32),
    torch.tensor([-1, 0, 18, 0, -48, 0, 32], dtype=torch.float32),
    torch.tensor([0, -7, 0, 56, 0, -112, 0, 64], dtype=torch.float32),
    torch.tensor([1, 0, -32, 0, 160, 0, -256, 0, 128], dtype=torch.float32),
    torch.tensor([0, 9, 0, -120, 0, 432, 0, -576, 0, 256], dtype=torch.float32),
]


def test_power_series_from_roots():
    torch.testing.assert_close(
        beignet.polynomial.trim_power_series(
            beignet.polynomial.power_series_from_roots(
                torch.tensor([]),
            ),
            tolerance=0.000001,
        ),
        torch.tensor([1], dtype=torch.float32),
    )

    for index in range(1, 5):
        output = torch.linspace(-torch.pi, 0, 2 * index + 1)

        output = output[1::2]

        output = torch.cos(output)

        output = beignet.polynomial.power_series_from_roots(output)

        output = output * 2 ** (index - 1)

        torch.testing.assert_close(
            beignet.polynomial.trim_power_series(
                output,
                tolerance=0.000001,
            ),
            beignet.polynomial.trim_power_series(
                polynomial_coefficients[index],
                tolerance=0.000001,
            ),
        )
