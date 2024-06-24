import beignet.orthax
import numpy
import numpy.testing

chebcoefficients = [
    [1],
    [0, 1],
    [-1, 0, 2],
    [0, -3, 0, 4],
    [1, 0, -8, 0, 8],
    [0, 5, 0, -20, 0, 16],
    [-1, 0, 18, 0, -48, 0, 32],
    [0, -7, 0, 56, 0, -112, 0, 64],
    [1, 0, -32, 0, 160, 0, -256, 0, 128],
    [0, 9, 0, -120, 0, 432, 0, -576, 0, 256],
]


class TestInterpolate:
    def f(self, x):
        return x * (x - 1) * (x - 2)

    def test_raises(self):
        numpy.testing.assert_raises(
            ValueError, beignet.orthax.chebinterpolate, self.f, -1
        )

    def test_dimensions(self):
        for deg in range(1, 5):
            numpy.testing.assert_(
                beignet.orthax.chebinterpolate(self.f, deg).shape == (deg + 1,)
            )

    def test_approximation(self):
        def powx(x, p):
            return x**p

        x = numpy.linspace(-1, 1, 10)
        for deg in range(0, 10):
            for p in range(0, deg + 1):
                c = beignet.orthax.chebinterpolate(powx, deg, (p,))
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.chebval(x, c), powx(x, p), decimal=12
                )
