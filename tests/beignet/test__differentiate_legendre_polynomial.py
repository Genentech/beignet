import beignet
import pytest
import torch


def test_differentiate_legendre_polynomial():
    with pytest.raises(TypeError):
        beignet.differentiate_legendre_polynomial(
            torch.tensor([0.0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        beignet.differentiate_legendre_polynomial(
            torch.tensor([0.0]),
            order=-1,
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.trim_legendre_polynomial_coefficients(
                beignet.differentiate_legendre_polynomial(
                    torch.tensor([0.0] * i + [1.0]),
                    order=0,
                ),
                tol=0.000001,
            ),
            beignet.trim_legendre_polynomial_coefficients(
                torch.tensor([0.0] * i + [1.0]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            torch.testing.assert_close(
                beignet.trim_legendre_polynomial_coefficients(
                    beignet.differentiate_legendre_polynomial(
                        beignet.integrate_legendre_polynomial(
                            torch.tensor([0.0] * i + [1.0]),
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                beignet.trim_legendre_polynomial_coefficients(
                    torch.tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            torch.testing.assert_close(
                beignet.trim_legendre_polynomial_coefficients(
                    beignet.differentiate_legendre_polynomial(
                        beignet.integrate_legendre_polynomial(
                            torch.tensor([0.0] * i + [1.0]),
                            order=j,
                            scale=2,
                        ),
                        order=j,
                        scale=0.5,
                    ),
                    tol=0.000001,
                ),
                beignet.trim_legendre_polynomial_coefficients(
                    torch.tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    coefficients = torch.rand(3, 4)

    target = []

    for input in coefficients.T:
        target = [
            *target,
            beignet.differentiate_legendre_polynomial(
                input,
            ),
        ]

    torch.testing.assert_close(
        beignet.differentiate_legendre_polynomial(
            coefficients,
            axis=0,
        ),
        torch.vstack(target).T,
    )

    target = []

    for input in coefficients:
        target = [
            *target,
            beignet.differentiate_legendre_polynomial(
                input,
            ),
        ]

    # torch.testing.assert_close(
    #     beignet.legder(
    #         coefficients,
    #         axis=1,
    #     ),
    #     torch.vstack(target),
    # )

    torch.testing.assert_close(
        beignet.differentiate_legendre_polynomial(
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            order=4,
        ),
        torch.tensor([0.0]),
    )
