import beignet.polynomial
import pytest
import torch


def test_chebder():
    with pytest.raises(TypeError):
        beignet.polynomial.chebder(
            torch.tensor([0.0]),
            torch.tensor([0.5]),
        )

    with pytest.raises(ValueError):
        beignet.polynomial.chebder(
            torch.tensor([0.0]),
            order=torch.tensor([-1.0]),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.polynomial.chebtrim(
                beignet.polynomial.chebder(
                    torch.tensor([0.0] * i + [1.0]),
                    order=torch.tensor([0.0]),
                ),
                tol=0.000001,
            ),
            beignet.polynomial.chebtrim(
                torch.tensor([0.0] * i + [1.0]),
                tol=0.000001,
            ),
        )

    # for i in range(5):
    #     for j in range(2, 5):
    #         assert_close(
    #             chebtrim(
    #                 chebder(
    #                     chebint(
    #                         tensor([0.0] * i + [1.0]),
    #                         order=j
    #                     ),
    #                     order=j,
    #                 ),
    #                 tol=0.000001,
    #             ),
    #             chebtrim(
    #                 tensor([0.0] * i + [1.0]),
    #                 tol=0.000001,
    #             ),
    #         )

    for i in range(5):
        for j in range(2, 5):
            torch.testing.assert_close(
                beignet.polynomial.chebtrim(
                    beignet.polynomial.chebder(
                        beignet.polynomial.chebint(
                            torch.tensor([0.0] * i + [1.0]),
                            order=j,
                            scale=2.0,
                        ),
                        order=j,
                        scale=0.5,
                    ),
                    tol=0.000001,
                ),
                beignet.polynomial.chebtrim(
                    torch.tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    input = torch.rand(3, 4)

    torch.testing.assert_close(
        beignet.polynomial.chebder(
            input,
            axis=0,
        ),
        torch.vstack([beignet.polynomial.chebder(c) for c in input.T]).T,
    )

    torch.testing.assert_close(
        beignet.polynomial.chebder(
            input,
            axis=1,
        ),
        torch.vstack([beignet.polynomial.chebder(c) for c in input]),
    )
