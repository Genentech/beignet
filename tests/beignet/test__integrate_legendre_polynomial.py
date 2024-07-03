import beignet
import pytest
import torch


def test_integrate_legendre_polynomial():
    with pytest.raises(TypeError):
        beignet.integrate_legendre_polynomial(
            torch.tensor([0]),
            0.5,
        )

    with pytest.raises(ValueError):
        beignet.integrate_legendre_polynomial(
            torch.tensor([0]),
            -1,
        )

    with pytest.raises(ValueError):
        beignet.integrate_legendre_polynomial(
            torch.tensor([0]),
            1,
            torch.tensor([0, 0]),
        )

    with pytest.raises(ValueError):
        beignet.integrate_legendre_polynomial(
            torch.tensor([0]),
            lower_bound=[0],
        )

    with pytest.raises(ValueError):
        beignet.integrate_legendre_polynomial(
            torch.tensor([0]),
            scale=[0],
        )

    with pytest.raises(TypeError):
        beignet.integrate_legendre_polynomial(
            torch.tensor([0]),
            axis=0.5,
        )

    for i in range(2, 5):
        output = beignet.integrate_legendre_polynomial(
            torch.tensor([0.0]),
            order=i,
            k=[0.0] * (i - 2) + [1.0],
        )
        torch.testing.assert_close(
            beignet.trim_legendre_polynomial_coefficients(
                output,
                tol=0.000001,
            ),
            torch.tensor([0.0, 1.0]),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.trim_legendre_polynomial_coefficients(
                beignet.legendre_polynomial_to_polynomial(
                    beignet.integrate_legendre_polynomial(
                        beignet.polynomial_to_legendre_polynomial(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                    )
                ),
                tol=0.000001,
            ),
            beignet.trim_legendre_polynomial_coefficients(
                torch.tensor([i] + [0.0] * i + [1 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.evaluate_legendre_polynomial(
                torch.tensor([-1]),
                beignet.integrate_legendre_polynomial(
                    beignet.polynomial_to_legendre_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                    ),
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            torch.tensor([i], dtype=torch.get_default_dtype()),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.trim_legendre_polynomial_coefficients(
                beignet.legendre_polynomial_to_polynomial(
                    beignet.integrate_legendre_polynomial(
                        beignet.polynomial_to_legendre_polynomial(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                        scale=2,
                    )
                ),
                tol=0.000001,
            ),
            beignet.trim_legendre_polynomial_coefficients(
                torch.tensor([i] + [0.0] * i + [2 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for _ in range(j):
                target = beignet.integrate_legendre_polynomial(
                    target,
                    order=1,
                )

            torch.testing.assert_close(
                beignet.trim_legendre_polynomial_coefficients(
                    beignet.integrate_legendre_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                beignet.trim_legendre_polynomial_coefficients(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = beignet.integrate_legendre_polynomial(
                    target,
                    order=1,
                    k=[k],
                )

            torch.testing.assert_close(
                beignet.trim_legendre_polynomial_coefficients(
                    beignet.integrate_legendre_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                    ),
                    tol=0.000001,
                ),
                beignet.trim_legendre_polynomial_coefficients(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = beignet.integrate_legendre_polynomial(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            torch.testing.assert_close(
                beignet.trim_legendre_polynomial_coefficients(
                    beignet.integrate_legendre_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                        lower_bound=-1,
                    ),
                    tol=0.000001,
                ),
                beignet.trim_legendre_polynomial_coefficients(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = beignet.integrate_legendre_polynomial(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )

            torch.testing.assert_close(
                beignet.trim_legendre_polynomial_coefficients(
                    beignet.integrate_legendre_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                        scale=2,
                    ),
                    tol=0.000001,
                ),
                beignet.trim_legendre_polynomial_coefficients(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = torch.rand(3, 4)

    torch.testing.assert_close(
        beignet.integrate_legendre_polynomial(c2d, axis=0),
        torch.vstack([beignet.integrate_legendre_polynomial(c) for c in c2d.T]).T,
    )

    target = [beignet.integrate_legendre_polynomial(c) for c in c2d]

    target = torch.vstack(target)

    torch.testing.assert_close(
        beignet.integrate_legendre_polynomial(
            c2d,
            axis=1,
        ),
        target,
    )

    target = [beignet.integrate_legendre_polynomial(c, k=3) for c in c2d]

    target = torch.vstack(target)

    torch.testing.assert_close(
        beignet.integrate_legendre_polynomial(
            c2d,
            k=3,
            axis=1,
        ),
        target,
    )

    torch.testing.assert_close(
        beignet.integrate_legendre_polynomial(
            torch.tensor([1, 2, 3]),
            order=0,
        ),
        torch.tensor([1, 2, 3]),
    )
