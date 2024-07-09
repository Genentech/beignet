import beignet
import pytest
import torch


def test_integrate_physicists_hermite_polynomial():
    with pytest.raises(TypeError):
        beignet.integrate_physicists_hermite_polynomial(
            torch.tensor([0.0]),
            order=0.5,
        )

    with pytest.raises(ValueError):
        beignet.integrate_physicists_hermite_polynomial(
            torch.tensor([0]),
            order=-1,
        )

    with pytest.raises(ValueError):
        beignet.integrate_physicists_hermite_polynomial(
            torch.tensor([0]),
            order=1,
            k=torch.tensor([0, 0]),
        )

    with pytest.raises(ValueError):
        beignet.integrate_physicists_hermite_polynomial(
            torch.tensor([0]),
            lower_bound=[0],
        )

    with pytest.raises(ValueError):
        beignet.integrate_physicists_hermite_polynomial(
            torch.tensor([0]),
            scale=[0],
        )

    with pytest.raises(TypeError):
        beignet.integrate_physicists_hermite_polynomial(
            torch.tensor([0]),
            axis=0.5,
        )

    for i in range(2, 5):
        torch.testing.assert_close(
            beignet.trim_physicists_hermite_polynomial_coefficients(
                beignet.integrate_physicists_hermite_polynomial(
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
            beignet.trim_physicists_hermite_polynomial_coefficients(
                beignet.physicists_hermite_polynomial_to_polynomial(
                    beignet.integrate_physicists_hermite_polynomial(
                        beignet.polynomial_to_physicists_hermite_polynomial(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                    )
                ),
                tol=0.000001,
            ),
            beignet.trim_physicists_hermite_polynomial_coefficients(
                torch.tensor([i] + [0.0] * i + [1.0 / (i + 1)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        torch.testing.assert_close(
            beignet.evaluate_physicists_hermite_polynomial(
                torch.tensor([-1.0]),
                beignet.integrate_physicists_hermite_polynomial(
                    beignet.polynomial_to_physicists_hermite_polynomial(
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
            beignet.trim_physicists_hermite_polynomial_coefficients(
                beignet.physicists_hermite_polynomial_to_polynomial(
                    beignet.integrate_physicists_hermite_polynomial(
                        beignet.polynomial_to_physicists_hermite_polynomial(
                            torch.tensor([0.0] * i + [1.0]),
                        ),
                        order=1,
                        k=[i],
                        scale=2,
                    ),
                ),
                tol=0.000001,
            ),
            beignet.trim_physicists_hermite_polynomial_coefficients(
                torch.tensor([i] + [0.0] * i + [2.0 / (i + 1.0)]),
                tol=0.000001,
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            target = torch.tensor([0.0] * i + [1.0])[:]

            for _ in range(j):
                target = beignet.integrate_physicists_hermite_polynomial(
                    target,
                    order=1,
                )

            torch.testing.assert_close(
                beignet.trim_physicists_hermite_polynomial_coefficients(
                    beignet.integrate_physicists_hermite_polynomial(
                        torch.tensor([0.0] * i + [1.0]),
                        order=j,
                    ),
                    tol=0.000001,
                ),
                beignet.trim_physicists_hermite_polynomial_coefficients(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = torch.tensor([0.0] * i + [1.0])

            target = pol[:]

            for k in range(j):
                target = beignet.integrate_physicists_hermite_polynomial(
                    target, order=1, k=[k]
                )

            torch.testing.assert_close(
                beignet.trim_physicists_hermite_polynomial_coefficients(
                    beignet.integrate_physicists_hermite_polynomial(
                        pol,
                        order=j,
                        k=list(range(j)),
                    ),
                    tol=0.000001,
                ),
                beignet.trim_physicists_hermite_polynomial_coefficients(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = torch.tensor([0.0] * i + [1.0])
            target = pol[:]
            for k in range(j):
                target = beignet.integrate_physicists_hermite_polynomial(
                    target,
                    order=1,
                    k=[k],
                    lower_bound=-1,
                )

            torch.testing.assert_close(
                beignet.trim_physicists_hermite_polynomial_coefficients(
                    beignet.integrate_physicists_hermite_polynomial(
                        pol,
                        order=j,
                        k=list(range(j)),
                        lower_bound=-1,
                    ),
                    tol=0.000001,
                ),
                beignet.trim_physicists_hermite_polynomial_coefficients(
                    target,
                    tol=0.000001,
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            pol = torch.tensor([0.0] * i + [1.0])
            target = pol[:]
            for k in range(j):
                target = beignet.integrate_physicists_hermite_polynomial(
                    target,
                    order=1,
                    k=[k],
                    scale=2,
                )

            # torch.testing.assert_close(
            #     beignet.hermtrim(
            #         beignet.hermint(
            #             pol,
            #             order=j,
            #             k=list(range(j)),
            #             scale=2,
            #         ),
            #         tol=0.000001,
            #     ),
            #     beignet.hermtrim(
            #         target,
            #         tol=0.000001,
            #     ),
            # )

    c2d = torch.rand(3, 4)

    target = torch.vstack(
        [beignet.integrate_physicists_hermite_polynomial(c) for c in c2d.T]
    ).T

    torch.testing.assert_close(
        beignet.integrate_physicists_hermite_polynomial(
            c2d,
            axis=0,
        ),
        target,
    )

    target = torch.vstack(
        [beignet.integrate_physicists_hermite_polynomial(c) for c in c2d]
    )

    torch.testing.assert_close(
        beignet.integrate_physicists_hermite_polynomial(
            c2d,
            axis=1,
        ),
        target,
    )

    target = torch.vstack(
        [beignet.integrate_physicists_hermite_polynomial(c, k=3) for c in c2d]
    )

    torch.testing.assert_close(
        beignet.integrate_physicists_hermite_polynomial(
            c2d,
            k=3,
            axis=1,
        ),
        target,
    )
