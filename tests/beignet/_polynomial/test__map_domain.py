import beignet.polynomial
import beignet.polynomial.__map_domain
import numpy


def test__map_domain():
    numpy.testing.assert_almost_equal(
        beignet.polynomial.__map_domain._map_domain(
            [0, 4],
            [0, 4],
            [1, 3],
        ),
        [1, 3],
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial.__map_domain._map_domain(
            [0 - 1j, 2 + 1j],
            [0 - 1j, 2 + 1j],
            [-2, 2],
        ),
        [-2, 2],
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial.__map_domain._map_domain(
            numpy.array([[0, 4], [0, 4]]),
            [0, 4],
            [1, 3],
        ),
        numpy.array([[1, 3], [1, 3]]),
    )

    class Foo(numpy.ndarray):
        pass

    numpy.testing.assert_(
        isinstance(
            beignet.polynomial.__map_domain._map_domain(
                numpy.array([[0, 4], [0, 4]]).view(Foo), [0, 4], [1, 3]
            ),
            Foo,
        )
    )
