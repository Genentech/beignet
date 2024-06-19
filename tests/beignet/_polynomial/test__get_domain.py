import beignet._polynomial.__get_domain
import beignet.polynomial
import numpy
import torch


def test__get_domain():
    numpy.testing.assert_almost_equal(
        beignet._polynomial.__get_domain._get_domain(
            torch.tensor(
                [
                    +1,
                    +2,
                    +3,
                    -1,
                ],
            )
        ),
        [-1, +3],
    )

    numpy.testing.assert_almost_equal(
        beignet._polynomial.__get_domain._get_domain(
            torch.tensor(
                [
                    1.0 + 1.0j,
                    1.0 - 1.0j,
                    0.0 + 0.0j,
                    2.0 + 0.0j,
                ],
            ),
        ),
        [
            0.0 - 1.0j,
            2.0 + 1.0j,
        ],
    )
