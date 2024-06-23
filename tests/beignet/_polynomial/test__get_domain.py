import beignet.polynomial
import beignet.polynomial.__get_domain
import torch


def test__get_domain():
    torch.testing.assert_close(
        beignet.polynomial.__get_domain._get_domain(
            torch.tensor(
                [
                    +1,
                    +2,
                    +3,
                    -1,
                ],
            )
        ),
        torch.tensor(
            [
                -1,
                +3,
            ],
            dtype=torch.float64,
        ),
    )

    torch.testing.assert_close(
        beignet.polynomial.__get_domain._get_domain(
            torch.tensor(
                [
                    1.0 + 1.0j,
                    1.0 - 1.0j,
                    0.0 + 0.0j,
                    2.0 + 0.0j,
                ],
            ),
        ),
        torch.tensor(
            [
                0.0 - 1.0j,
                2.0 + 1.0j,
            ],
        ),
    )
