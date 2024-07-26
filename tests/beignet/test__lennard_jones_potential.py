import beignet
import torch
import torch.testing


def test__lennard_jones_potential():
    x = beignet.lennard_jones_potential(
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        ),
        1.0,
        1.0,
    )

    print(x)

    torch.testing.assert_close(
        beignet.lennard_jones_potential(
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
            ),
            1.0,
            1.0,
        ),
        torch.tensor(
            [
                [+0.0000000000000000, -0.0615234375000000],
                [-0.0054794428870082, -0.0009763240814209],
            ],
        ),
    )
