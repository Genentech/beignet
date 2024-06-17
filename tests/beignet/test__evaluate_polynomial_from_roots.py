import math

import beignet
import pytest
import torch


def test_evaluate_polynomial_from_roots():
    with pytest.raises(ValueError):
        beignet.evaluate_polynomial_from_roots(
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            tensor=False,
        )

    output = beignet.evaluate_polynomial_from_roots(
        torch.tensor([]),
        torch.tensor([1.0]),
    )

    assert math.prod(output.shape) == 0

    assert output.shape == (0,)

    output = beignet.evaluate_polynomial_from_roots(
        torch.tensor([]),
        torch.tensor([[1.0] * 5]),
    )

    assert math.prod(output.shape) == 0

    assert output.shape == (5, 0)

    torch.testing.assert_close(
        beignet.evaluate_polynomial_from_roots(
            torch.tensor([1.0]),
            torch.tensor([1.0]),
        ),
        torch.tensor([0.0]),
    )

    output = beignet.evaluate_polynomial_from_roots(
        torch.tensor([1.0]),
        torch.ones([3, 3]),
    )

    assert output.shape == (3, 1)

    input = torch.linspace(-1, 1, 50)

    evaluations = []

    for i in range(5):
        evaluations = [*evaluations, input**i]

    for i in range(1, 5):
        target = evaluations[i]

        torch.testing.assert_close(
            beignet.evaluate_polynomial_from_roots(
                input,
                torch.tensor([0.0] * i),
            ),
            target,
        )

    torch.testing.assert_close(
        beignet.evaluate_polynomial_from_roots(
            input,
            torch.tensor([-1.0, 0.0, 1.0]),
        ),
        input * (input - 1.0) * (input + 1.0),
    )

    for i in range(3):
        shape = (2,) * i

        input = torch.zeros(shape)

        output = beignet.evaluate_polynomial_from_roots(
            input,
            torch.tensor([1.0]),
        )

        assert output.shape == shape

        output = beignet.evaluate_polynomial_from_roots(
            input,
            torch.tensor([1.0, 0.0]),
        )

        assert output.shape == shape

        output = beignet.evaluate_polynomial_from_roots(
            input,
            torch.tensor([1.0, 0.0, 0.0]),
        )

        assert output.shape == shape

    ptest = torch.tensor([15.0, 2.0, -16.0, -2.0, 1.0])

    r = beignet.polynomial_roots(ptest)

    torch.testing.assert_close(
        beignet.evaluate_polynomial(
            input,
            ptest,
        ),
        beignet.evaluate_polynomial_from_roots(
            input,
            r,
        ),
    )

    x = torch.arange(-3, 2)

    r = torch.randint(-5, 5, (3, 5)).to(torch.float64)

    target = torch.empty(r.shape[1:])

    for j in range(math.prod(target.shape)):
        target[j] = beignet.evaluate_polynomial_from_roots(
            x[j],
            r[:, j],
        )

    torch.testing.assert_close(
        beignet.evaluate_polynomial_from_roots(
            x,
            r,
            tensor=False,
        ),
        target,
    )

    x = torch.vstack([x, 2 * x])

    target = torch.empty(r.shape[1:] + x.shape)

    for j in range(r.shape[1]):
        for k in range(x.shape[0]):
            target[j, k, :] = beignet.evaluate_polynomial_from_roots(
                x[k],
                r[:, j],
            )

    torch.testing.assert_close(
        beignet.evaluate_polynomial_from_roots(
            x,
            r,
            tensor=True,
        ),
        target,
    )
