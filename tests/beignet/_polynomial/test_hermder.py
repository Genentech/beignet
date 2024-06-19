import beignet.polynomial
import numpy
import torch


def test_hermder():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.hermder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.hermder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        torch.testing.assert_close(
            beignet.polynomial.hermtrim(
                beignet.polynomial.hermder(tgt, m=0), tolerance=1e-6
            ),
            beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermtrim(
                    beignet.polynomial.hermder(
                        beignet.polynomial.hermint(tgt, m=j), m=j
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                beignet.polynomial.hermtrim(
                    beignet.polynomial.hermder(
                        beignet.polynomial.hermint(tgt, m=j, scl=2), m=j, scl=0.5
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.hermtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.hermder(c) for c in c2d.T]).T
    numpy.testing.assert_almost_equal(beignet.polynomial.hermder(c2d, axis=0), tgt)

    tgt = numpy.vstack([beignet.polynomial.hermder(c) for c in c2d])
    numpy.testing.assert_almost_equal(beignet.polynomial.hermder(c2d, axis=1), tgt)
