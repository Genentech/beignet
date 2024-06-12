import beignet.polynomial.polyutils
import numpy
import numpy.testing


class TestMisc:
    def test_trimseq(self):
        for _ in range(5):
            tgt = [1]
            res = beignet.polynomial.polyutils.trimseq([1] + [0] * 5)
            numpy.testing.assert_equal(res, tgt)

    def test_as_series(self):
        # check exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polyutils.as_series, [[]]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polyutils.as_series, [[[1, 2]]]
        )
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polyutils.as_series, [[1], ["a"]]
        )
        # check common types
        types = ["i", "d", "O"]
        for i in range(len(types)):
            for j in range(i):
                ci = numpy.ones(1, types[i])
                cj = numpy.ones(1, types[j])
                [resi, resj] = beignet.polynomial.polyutils.as_series([ci, cj])
                numpy.testing.assert_(resi.dtype.char == resj.dtype.char)
                numpy.testing.assert_(resj.dtype.char == types[i])

    def test_trimcoef(self):
        coef = [2, -1, 1, 0]
        # Test exceptions
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polyutils.trimcoef, coef, -1
        )
        # Test results
        numpy.testing.assert_equal(
            beignet.polynomial.polyutils.trimcoef(coef), coef[:-1]
        )
        numpy.testing.assert_equal(
            beignet.polynomial.polyutils.trimcoef(coef, 1), coef[:-3]
        )
        numpy.testing.assert_equal(beignet.polynomial.polyutils.trimcoef(coef, 2), [0])

    def test_vander_nd_exception(self):
        # n_dims != len(points)
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polyutils._vander_nd, (), (1, 2, 3), [90]
        )
        # n_dims != len(degrees)
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polyutils._vander_nd, (), (), [90.65]
        )
        # n_dims == 0
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polyutils._vander_nd, (), (), []
        )

    def test_div_zerodiv(self):
        # c2[-1] == 0
        numpy.testing.assert_raises(
            ZeroDivisionError,
            beignet.polynomial.polyutils._div,
            beignet.polynomial.polyutils._div,
            (1, 2, 3),
            [0],
        )

    def test_pow_too_large(self):
        # power > maxpower
        numpy.testing.assert_raises(
            ValueError, beignet.polynomial.polyutils._pow, (), [1, 2, 3], 5, 4
        )


class TestDomain:
    def test_getdomain(self):
        # test for real values
        x = [1, 10, 3, -1]
        tgt = [-1, 10]
        res = beignet.polynomial.polyutils.getdomain(x)
        numpy.testing.assert_almost_equal(res, tgt)

        # test for complex values
        x = [1 + 1j, 1 - 1j, 0, 2]
        tgt = [-1j, 2 + 1j]
        res = beignet.polynomial.polyutils.getdomain(x)
        numpy.testing.assert_almost_equal(res, tgt)

    def test_mapdomain(self):
        # test for real values
        dom1 = [0, 4]
        dom2 = [1, 3]
        tgt = dom2
        res = beignet.polynomial.polyutils.mapdomain(dom1, dom1, dom2)
        numpy.testing.assert_almost_equal(res, tgt)

        # test for complex values
        dom1 = [0 - 1j, 2 + 1j]
        dom2 = [-2, 2]
        tgt = dom2
        x = dom1
        res = beignet.polynomial.polyutils.mapdomain(x, dom1, dom2)
        numpy.testing.assert_almost_equal(res, tgt)

        # test for multidimensional arrays
        dom1 = [0, 4]
        dom2 = [1, 3]
        tgt = numpy.array([dom2, dom2])
        x = numpy.array([dom1, dom1])
        res = beignet.polynomial.polyutils.mapdomain(x, dom1, dom2)
        numpy.testing.assert_almost_equal(res, tgt)

        # test that subtypes are preserved.
        class MyNDArray(numpy.ndarray):
            pass

        dom1 = [0, 4]
        dom2 = [1, 3]
        x = numpy.array([dom1, dom1]).view(MyNDArray)
        res = beignet.polynomial.polyutils.mapdomain(x, dom1, dom2)
        numpy.testing.assert_(isinstance(res, MyNDArray))

    def test_mapparms(self):
        # test for real values
        dom1 = [0, 4]
        dom2 = [1, 3]
        tgt = [1, 0.5]
        res = beignet.polynomial.polyutils.mapparms(dom1, dom2)
        numpy.testing.assert_almost_equal(res, tgt)

        # test for complex values
        dom1 = [0 - 1j, 2 + 1j]
        dom2 = [-2, 2]
        tgt = [-1 + 1j, 1 - 1j]
        res = beignet.polynomial.polyutils.mapparms(dom1, dom2)
        numpy.testing.assert_almost_equal(res, tgt)