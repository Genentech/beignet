import beignet.polynomial
import pytest


def test__vander_nd():
    with pytest.raises(ValueError):
        beignet.polynomial._vander_nd(
            (),
            (1, 2, 3),
            [90],
        )

    with pytest.raises(ValueError):
        beignet.polynomial._vander_nd(
            (),
            (),
            [90.65],
        )

    with pytest.raises(ValueError):
        beignet.polynomial._vander_nd(
            (),
            (),
            [],
        )
