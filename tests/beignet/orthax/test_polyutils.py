import beignet.orthax.polyutils
import numpy
import numpy.testing


def test_trimseq():
    for _ in range(5):
        numpy.testing.assert_equal(beignet.orthax.polyutils.trimseq([1] + [0] * 5), [1])


def test_trimcoef():
    coef = numpy.array([2, -1, 1, 0])

    numpy.testing.assert_raises(ValueError, beignet.orthax.polyutils.trimcoef, coef, -1)

    numpy.testing.assert_equal(beignet.orthax.polyutils.trimcoef(coef), coef[:-1])
    numpy.testing.assert_equal(beignet.orthax.polyutils.trimcoef(coef, 1), coef[:-3])
    numpy.testing.assert_equal(
        beignet.orthax.polyutils.trimcoef(coef, 2), numpy.array([0])
    )


def test_vander_nd_exception():
    numpy.testing.assert_raises(
        ValueError, beignet.orthax.polyutils._vander_nd, (), (1, 2, 3), [90]
    )

    numpy.testing.assert_raises(
        ValueError, beignet.orthax.polyutils._vander_nd, (), (), [90.65]
    )

    numpy.testing.assert_raises(
        ValueError, beignet.orthax.polyutils._vander_nd, (), (), []
    )


def test_pow_too_large():
    numpy.testing.assert_raises(
        ValueError, beignet.orthax.polyutils._pow, (), [1, 2, 3], 5, 4
    )


def test_getdomain():
    x = [1, 10, 3, -1]
    tgt = [-1, 10]
    res = beignet.orthax.polyutils.getdomain(x)
    numpy.testing.assert_array_equal(res, tgt)

    x = [1 + 1j, 1 - 1j, 0, 2]
    tgt = [-1j, 2 + 1j]
    res = beignet.orthax.polyutils.getdomain(x)
    numpy.testing.assert_array_equal(res, tgt)


def test_mapdomain():
    dom1 = [0, 4]
    dom2 = [1, 3]
    tgt = dom2
    res = beignet.orthax.polyutils.mapdomain(dom1, dom1, dom2)
    numpy.testing.assert_array_equal(res, tgt)

    dom1 = [0 - 1j, 2 + 1j]
    dom2 = [-2, 2]
    tgt = dom2
    x = dom1
    res = beignet.orthax.polyutils.mapdomain(x, dom1, dom2)
    numpy.testing.assert_array_equal(res, tgt)

    dom1 = [0, 4]
    dom2 = [1, 3]
    tgt = numpy.array([dom2, dom2])
    x = numpy.array([dom1, dom1])
    res = beignet.orthax.polyutils.mapdomain(x, dom1, dom2)
    numpy.testing.assert_array_equal(res, tgt)


def test_mapparms():
    numpy.testing.assert_array_equal(
        beignet.orthax.polyutils.mapparms([0, 4], [1, 3]), [1, 0.5]
    )

    numpy.testing.assert_array_equal(
        beignet.orthax.polyutils.mapparms([0 - 1j, 2 + 1j], [-2, 2]), [-1 + 1j, 1 - 1j]
    )
