import beignet.polynomial
import pytest
import torch


def test_chebint():
    with pytest.raises(TypeError):
        beignet.polynomial.chebint(
            torch.tensor([0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        beignet.polynomial.chebint(
            torch.tensor([0]),
            order=-1,
        )

    with pytest.raises(ValueError):
        beignet.polynomial.chebint(
            torch.tensor([0.0]),
            order=1,
            k=torch.tensor([0.0, 0.0]),
        )

    with pytest.raises(TypeError):
        beignet.polynomial.chebint(
            torch.tensor([0]),
            axis=0.5,
        )

    for i in range(2, 5):
        torch.testing.assert_close(
            beignet.polynomial.chebtrim(
                beignet.polynomial.chebint(
                    torch.tensor([0.0]),
                    order=i,
                    k=torch.tensor([0.0] * (i - 2) + [1.0]),
                ),
                tol=0.000001,
            ),
            torch.tensor([0.0, 1.0]),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.polynomial.chebtrim(
                beignet.polynomial.cheb2poly(
                    beignet.polynomial.chebint(
                        beignet.polynomial.poly2cheb(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                    ),
                ),
                tol=0.000001,
            ),
            beignet.polynomial.chebtrim(
                torch.tensor([i] + [0] * i + [1 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.polynomial.chebval(
                torch.tensor([-1]),
                beignet.polynomial.chebint(
                    beignet.polynomial.poly2cheb(
                        torch.tensor([0.0] * i + [1.0]),
                    ),
                    order=1,
                    k=[i],
                    lower_bound=-1,
                ),
            ),
            i,
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.polynomial.chebtrim(
                beignet.polynomial.cheb2poly(
                    beignet.polynomial.chebint(
                        beignet.polynomial.poly2cheb(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                        scale=2,
                    )
                ),
                tol=0.000001,
            ),
            beignet.polynomial.chebtrim(
                torch.tensor([i] + [0] * i + [2 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            input = torch.tensor([0.0] * i + [1.0])
            target = input[:]

            for _ in range(j):
                target = beignet.polynomial.chebint(
                    target,
                    order=1,
                )

            torch.testing.assert_close(
                beignet.polynomial.chebtrim(
                    beignet.polynomial.chebint(
                        input,
                        order=j,
                    ),
                    tol=0.000001,
                ),
                beignet.polynomial.chebtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            input = torch.tensor([0.0] * i + [1.0])

            target = input[:]

            for k in range(j):
                target = beignet.polynomial.chebint(
                    target,
                    order=1,
                    k=[k],
                )

            torch.testing.assert_close(
                beignet.polynomial.chebtrim(
                    beignet.polynomial.chebint(
                        input,
                        order=j,
                        k=list(range(j)),
                    ),
                    tol=0.000001,
                ),
                beignet.polynomial.chebtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            input = torch.tensor([0.0] * i + [1.0])

            target = input[:]

            for k in range(j):
                target = beignet.polynomial.chebint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            torch.testing.assert_close(
                beignet.polynomial.chebtrim(
                    beignet.polynomial.chebint(
                        input,
                        order=j,
                        k=list(range(j)),
                        lower_bound=-1,
                    ),
                    tol=0.000001,
                ),
                beignet.polynomial.chebtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            input = torch.tensor([0.0] * i + [1.0])

            target = input[:]

            for k in range(j):
                target = beignet.polynomial.chebint(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )

            torch.testing.assert_close(
                beignet.polynomial.chebtrim(
                    beignet.polynomial.chebint(
                        input,
                        order=j,
                        k=list(range(j)),
                        scale=2,
                    ),
                    tol=0.000001,
                ),
                beignet.polynomial.chebtrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = torch.rand(3, 4)

    torch.testing.assert_close(
        beignet.polynomial.chebint(
            c2d,
            axis=0,
        ),
        torch.vstack([beignet.polynomial.chebint(c) for c in c2d.T]).T,
    )

    torch.testing.assert_close(
        beignet.polynomial.chebint(
            c2d,
            axis=1,
        ),
        torch.vstack([beignet.polynomial.chebint(c) for c in c2d]),
    )

    torch.testing.assert_close(
        beignet.polynomial.chebint(
            c2d,
            k=3,
            axis=1,
        ),
        torch.vstack([beignet.polynomial.chebint(c, k=3) for c in c2d]),
    )
