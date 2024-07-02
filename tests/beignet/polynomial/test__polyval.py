import math

import torch
from beignet.polynomial import polyval


def test_polyval():
    output = polyval(
        torch.tensor([]),
        torch.tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    y = []

    input = torch.linspace(-1, 1, 50)

    for index in range(5):
        y = [
            *y,
            input**index,
        ]

    for index in range(5):
        torch.testing.assert_close(
            polyval(
                input,
                torch.tensor([0.0] * index + [1.0]),
            ),
            y[index],
        )

    torch.testing.assert_close(
        polyval(
            input,
            torch.tensor([0, -1, 0, 1]),
        ),
        input * (input**2 - 1),
    )

    for index in range(3):
        shape = (2,) * index

        input = torch.zeros(shape)

        output = polyval(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = polyval(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = polyval(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape
