import beignet
import pytest
import torch


def test_integrate_polynomial():
    with pytest.raises(TypeError):
        beignet.integrate_polynomial(
            torch.tensor([0.0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        beignet.integrate_polynomial(
            torch.tensor([0.0]),
            order=-1,
        )

    with pytest.raises(ValueError):
        beignet.integrate_polynomial(
            torch.tensor([0.0]),
            order=1,
            k=[0, 0],
        )

    with pytest.raises(ValueError):
        beignet.integrate_polynomial(
            torch.tensor([0.0]),
            lower_bound=[0],
        )

    with pytest.raises(ValueError):
        beignet.integrate_polynomial(
            torch.tensor([0.0]),
            scale=[0],
        )

    with pytest.raises(TypeError):
        beignet.integrate_polynomial(
            torch.tensor([0.0]),
            axis=0.5,
        )

    for i in range(2, 5):
        torch.testing.assert_close(
            beignet.trim_polynomial_coefficients(
                beignet.integrate_polynomial(
                    torch.tensor([0.0]),
                    order=i,
                    k=[0.0] * (i - 2) + [1.0],
                ),
            ),
            torch.tensor([0.0, 1.0]),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.trim_polynomial_coefficients(
                beignet.integrate_polynomial(
                    torch.tensor([0.0] * i + [1.0]),
                    order=1,
                    k=[i],
                ),
            ),
            beignet.trim_polynomial_coefficients(
                torch.tensor([i] + [0.0] * i + [1.0 / (i + 1.0)]),
            ),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.evaluate_polynomial(
                torch.tensor([-1.0]),
                beignet.integrate_polynomial(
                    torch.tensor([0.0] * i + [1.0]),
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            torch.tensor([i], dtype=torch.get_default_dtype()),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.trim_polynomial_coefficients(
                beignet.integrate_polynomial(
                    torch.tensor([0.0] * i + [1.0]),
                    order=1,
                    k=[i],
                    scale=2,
                ),
            ),
            beignet.trim_polynomial_coefficients(
                torch.tensor([i] + [0.0] * i + [2.0 / (i + 1.0)]),
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for _ in range(j):
                target = beignet.integrate_polynomial(
                    target,
                    order=1,
                )

            torch.testing.assert_close(
                beignet.trim_polynomial_coefficients(
                    beignet.integrate_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                    ),
                ),
                beignet.trim_polynomial_coefficients(
                    target,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = beignet.integrate_polynomial(
                    target,
                    order=1,
                    k=[k],
                )

            torch.testing.assert_close(
                beignet.trim_polynomial_coefficients(
                    beignet.integrate_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                    ),
                ),
                beignet.trim_polynomial_coefficients(
                    target,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = beignet.integrate_polynomial(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            torch.testing.assert_close(
                beignet.trim_polynomial_coefficients(
                    beignet.integrate_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                        lower_bound=-1,
                    ),
                ),
                beignet.trim_polynomial_coefficients(
                    target,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = beignet.integrate_polynomial(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )

            torch.testing.assert_close(
                beignet.trim_polynomial_coefficients(
                    beignet.integrate_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                        scale=2,
                    ),
                ),
                beignet.trim_polynomial_coefficients(
                    target,
                ),
            )

    c2d = torch.rand(3, 6)

    torch.testing.assert_close(
        beignet.integrate_polynomial(
            c2d,
            dim=0,
        ),
        torch.vstack([beignet.integrate_polynomial(c) for c in c2d.T]).T,
    )

    torch.testing.assert_close(
        beignet.integrate_polynomial(
            c2d,
            dim=1,
        ),
        torch.vstack([beignet.integrate_polynomial(c) for c in c2d]),
    )

    torch.testing.assert_close(
        beignet.integrate_polynomial(
            c2d,
            k=3,
            dim=1,
        ),
        torch.vstack([beignet.integrate_polynomial(c, k=3) for c in c2d]),
    )
