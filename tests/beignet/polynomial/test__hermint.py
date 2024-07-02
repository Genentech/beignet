import beignet.polynomial
import pytest
import torch


def test_hermint():
    with pytest.raises(TypeError):
        beignet.polynomial.hermint(
            torch.tensor([0.0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        beignet.polynomial.hermint(
            torch.tensor([0]),
            order=-1,
        )

    with pytest.raises(ValueError):
        beignet.polynomial.hermint(
            torch.tensor([0]),
            order=1,
            k=torch.tensor([0, 0]),
        )

    with pytest.raises(ValueError):
        beignet.polynomial.hermint(
            torch.tensor([0]),
            lower_bound=[0],
        )

    with pytest.raises(ValueError):
        beignet.polynomial.hermint(
            torch.tensor([0]),
            scale=[0],
        )

    with pytest.raises(TypeError):
        beignet.polynomial.hermint(
            torch.tensor([0]),
            axis=0.5,
        )

    for i in range(2, 5):
        torch.testing.assert_close(
            beignet.polynomial.hermtrim(
                beignet.polynomial.hermint(
                    torch.tensor([0.0]),
                    order=i,
                    k=([0.0] * (i - 2) + [1.0]),
                ),
                tol=0.000001,
            ),
            torch.tensor([0.0, 0.5]),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.polynomial.hermtrim(
                beignet.polynomial.herm2poly(
                    beignet.polynomial.hermint(
                        beignet.polynomial.poly2herm(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                    )
                ),
                tol=0.000001,
            ),
            beignet.polynomial.hermtrim(
                torch.tensor([i] + [0.0] * i + [1.0 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.polynomial.hermval(
                torch.tensor([-1.0]),
                beignet.polynomial.hermint(
                    beignet.polynomial.poly2herm(
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
            beignet.polynomial.hermtrim(
                beignet.polynomial.herm2poly(
                    beignet.polynomial.hermint(
                        beignet.polynomial.poly2herm(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                        scale=2,
                    ),
                ),
                tol=0.000001,
            ),
            beignet.polynomial.hermtrim(
                torch.tensor([i] + [0.0] * i + [2.0 / (i + 1.0)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for _ in range(j):
                target = beignet.polynomial.hermint(
                    target,
                    order=1,
                )

            torch.testing.assert_close(
                beignet.polynomial.hermtrim(
                    beignet.polynomial.hermint(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                beignet.polynomial.hermtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = torch.tensor([0.0] * i + [1.0])

            target = pol[:]

            for k in range(j):
                target = beignet.polynomial.hermint(target, order=1, k=[k])

            torch.testing.assert_close(
                beignet.polynomial.hermtrim(
                    beignet.polynomial.hermint(
                        pol,
                        order=j,
                        k=list(range(j)),
                    ),
                    tol=0.000001,
                ),
                beignet.polynomial.hermtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = torch.tensor([0.0] * i + [1.0])
            target = pol[:]
            for k in range(j):
                target = beignet.polynomial.hermint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            torch.testing.assert_close(
                beignet.polynomial.hermtrim(
                    beignet.polynomial.hermint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        lower_bound=-1,
                    ),
                    tol=0.000001,
                ),
                beignet.polynomial.hermtrim(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = torch.tensor([0.0] * i + [1.0])
            target = pol[:]
            for k in range(j):
                target = beignet.polynomial.hermint(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )

            torch.testing.assert_close(
                beignet.polynomial.hermtrim(
                    beignet.polynomial.hermint(
                        pol,
                        order=j,
                        k=list(range(j)),
                        scale=2,
                    ),
                    tol=0.000001,
                ),
                beignet.polynomial.hermtrim(
                    target,
                    tol=0.000001,
                ),
            )

    c2d = torch.rand(3, 4)

    target = torch.vstack([beignet.polynomial.hermint(c) for c in c2d.T]).T

    torch.testing.assert_close(
        beignet.polynomial.hermint(
            c2d,
            axis=0,
        ),
        target,
    )

    target = torch.vstack([beignet.polynomial.hermint(c) for c in c2d])

    torch.testing.assert_close(
        beignet.polynomial.hermint(
            c2d,
            axis=1,
        ),
        target,
    )

    target = torch.vstack([beignet.polynomial.hermint(c, k=3) for c in c2d])

    torch.testing.assert_close(
        beignet.polynomial.hermint(
            c2d,
            k=3,
            axis=1,
        ),
        target,
    )
