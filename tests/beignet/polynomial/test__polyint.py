import beignet.polynomial
import pytest
import torch


def test_polyint():
    with pytest.raises(TypeError):
        beignet.polynomial.polyint(
            torch.tensor([0.0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        beignet.polynomial.polyint(
            torch.tensor([0.0]),
            order=-1,
        )

    with pytest.raises(ValueError):
        beignet.polynomial.polyint(
            torch.tensor([0.0]),
            order=1,
            k=[0, 0],
        )

    with pytest.raises(ValueError):
        beignet.polynomial.polyint(
            torch.tensor([0.0]),
            lower_bound=[0],
        )

    with pytest.raises(ValueError):
        beignet.polynomial.polyint(
            torch.tensor([0.0]),
            scale=[0],
        )

    with pytest.raises(TypeError):
        beignet.polynomial.polyint(
            torch.tensor([0.0]),
            axis=0.5,
        )

    for i in range(2, 5):
        torch.testing.assert_close(
            beignet.polynomial.polytrim(
                beignet.polynomial.polyint(
                    torch.tensor([0.0]),
                    order=i,
                    k=[0.0] * (i - 2) + [1.0],
                ),
            ),
            torch.tensor([0.0, 1.0]),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.polynomial.polytrim(
                beignet.polynomial.polyint(
                    torch.tensor([0.0] * i + [1.0]),
                    order=1,
                    k=[i],
                ),
            ),
            beignet.polynomial.polytrim(
                torch.tensor([i] + [0.0] * i + [1.0 / (i + 1.0)]),
            ),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.polynomial.polyval(
                torch.tensor([-1.0]),
                beignet.polynomial.polyint(
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
            beignet.polynomial.polytrim(
                beignet.polynomial.polyint(
                    torch.tensor([0.0] * i + [1.0]),
                    order=1,
                    k=[i],
                    scale=2,
                ),
            ),
            beignet.polynomial.polytrim(
                torch.tensor([i] + [0.0] * i + [2.0 / (i + 1.0)]),
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for _ in range(j):
                target = beignet.polynomial.polyint(
                    target,
                    order=1,
                )

            torch.testing.assert_close(
                beignet.polynomial.polytrim(
                    beignet.polynomial.polyint(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                    ),
                ),
                beignet.polynomial.polytrim(
                    target,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = beignet.polynomial.polyint(
                    target,
                    order=1,
                    k=[k],
                )

            torch.testing.assert_close(
                beignet.polynomial.polytrim(
                    beignet.polynomial.polyint(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                    ),
                ),
                beignet.polynomial.polytrim(
                    target,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = beignet.polynomial.polyint(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            torch.testing.assert_close(
                beignet.polynomial.polytrim(
                    beignet.polynomial.polyint(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                        lower_bound=-1,
                    ),
                ),
                beignet.polynomial.polytrim(
                    target,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for k in range(j):
                target = beignet.polynomial.polyint(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )

            torch.testing.assert_close(
                beignet.polynomial.polytrim(
                    beignet.polynomial.polyint(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                        k=list(range(j)),
                        scale=2,
                    ),
                ),
                beignet.polynomial.polytrim(
                    target,
                ),
            )

    c2d = torch.rand(3, 6)

    torch.testing.assert_close(
        beignet.polynomial.polyint(
            c2d,
            axis=0,
        ),
        torch.vstack([beignet.polynomial.polyint(c) for c in c2d.T]).T,
    )

    torch.testing.assert_close(
        beignet.polynomial.polyint(
            c2d,
            axis=1,
        ),
        torch.vstack([beignet.polynomial.polyint(c) for c in c2d]),
    )

    torch.testing.assert_close(
        beignet.polynomial.polyint(
            c2d,
            k=3,
            axis=1,
        ),
        torch.vstack([beignet.polynomial.polyint(c, k=3) for c in c2d]),
    )
