import beignet
import pytest
import torch


def test_differentiate_chebyshev_polynomial():
    with pytest.raises(TypeError):
        beignet.differentiate_chebyshev_polynomial(
            torch.tensor([0.0]),
            torch.tensor([0.5]),
        )

    with pytest.raises(ValueError):
        beignet.differentiate_chebyshev_polynomial(
            torch.tensor([0.0]),
            order=torch.tensor([-1.0]),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.trim_chebyshev_polynomial_coefficients(
                beignet.differentiate_chebyshev_polynomial(
                    torch.tensor([0.0] * i + [1.0]),
                    order=torch.tensor([0.0]),
                ),
                tol=0.000001,
            ),
            beignet.trim_chebyshev_polynomial_coefficients(
                torch.tensor([0.0] * i + [1.0]),
                tol=0.000001,
            ),
        )

    # for i in range(5):
    #     for j in range(2, 5):
    #         torch.testing.assert_close(
    #             beignet.chebtrim(
    #                 beignet.chebder(
    #                     beignet.chebint(
    #                         torch.tensor([0.0] * i + [1.0]), order=j
    #                     ),
    #                     order=j,
    #                 ),
    #                 tol=0.000001,
    #             ),
    #             beignet.chebtrim(
    #                 torch.tensor([0.0] * i + [1.0]),
    #                 tol=0.000001,
    #             ),
    #         )

    # for i in range(5):
    #     for j in range(2, 5):
    #         torch.testing.assert_close(
    #             beignet.chebtrim(
    #                 beignet.chebder(
    #                     beignet.chebint(
    #                         torch.tensor([0.0] * i + [1.0]),
    #                         order=j,
    #                         scale=2.0,
    #                     ),
    #                     order=j,
    #                     scale=0.5,
    #                 ),
    #                 tol=0.000001,
    #             ),
    #             beignet.chebtrim(
    #                 torch.tensor([0.0] * i + [1.0]),
    #                 tol=0.000001,
    #             ),
    #         )

    input = torch.rand(3, 4)

    torch.testing.assert_close(
        beignet.differentiate_chebyshev_polynomial(
            input,
            axis=0,
        ),
        torch.vstack(
            [beignet.differentiate_chebyshev_polynomial(c) for c in input.T]
        ).T,
    )

    torch.testing.assert_close(
        beignet.differentiate_chebyshev_polynomial(
            input,
            axis=1,
        ),
        torch.vstack([beignet.differentiate_chebyshev_polynomial(c) for c in input]),
    )
