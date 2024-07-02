import beignet.polynomial
import pytest
import torch


def test_lagder():
    with pytest.raises(TypeError):
        beignet.polynomial.lagder(
            torch.tensor([0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        beignet.polynomial.lagder(
            torch.tensor([0]),
            order=-1,
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.polynomial.lagtrim(
                beignet.polynomial.lagder(
                    torch.tensor([0.0] * i + [1.0]),
                    order=0,
                ),
                tol=0.000001,
            ),
            beignet.polynomial.lagtrim(
                torch.tensor([0.0] * i + [1.0]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            torch.testing.assert_close(
                beignet.polynomial.lagtrim(
                    beignet.polynomial.lagder(
                        beignet.polynomial.lagint(
                            torch.tensor([0.0] * i + [1.0]),
                            order=j,
                        ),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                beignet.polynomial.lagtrim(
                    torch.tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            torch.testing.assert_close(
                beignet.polynomial.lagtrim(
                    beignet.polynomial.lagder(
                        beignet.polynomial.lagint(
                            torch.tensor([0.0] * i + [1.0]),
                            order=j,
                            scale=2,
                        ),
                        order=j,
                        scale=0.5,
                    ),
                    tol=0.000001,
                ),
                beignet.polynomial.lagtrim(
                    torch.tensor([0.0] * i + [1.0]),
                    tol=0.000001,
                ),
            )

    c2d = torch.rand(3, 4)

    torch.testing.assert_close(
        beignet.polynomial.lagder(c2d, axis=0),
        torch.vstack([beignet.polynomial.lagder(c) for c in c2d.T]).T,
    )

    torch.testing.assert_close(
        beignet.polynomial.lagder(
            c2d,
            axis=1,
        ),
        torch.vstack([beignet.polynomial.lagder(c) for c in c2d]),
    )
