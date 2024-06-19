import beignet.polynomial
import beignet.polynomial._hermder
import beignet.polynomial._hermint
import beignet.polynomial._hermtrim
import numpy
import torch


def test_hermder():
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._hermder.hermder, [0], 0.5
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._hermder.hermder, [0], -1
    )

    for i in range(5):
        tgt = [0] * i + [1]
        torch.testing.assert_close(
            beignet.polynomial._hermtrim.hermtrim(
                beignet.polynomial._hermder.hermder(tgt, m=0), tolerance=1e-6
            ),
            beignet.polynomial._hermtrim.hermtrim(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                beignet.polynomial._hermtrim.hermtrim(
                    beignet.polynomial._hermder.hermder(
                        beignet.polynomial._hermint.hermint(tgt, m=j), m=j
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial._hermtrim.hermtrim(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                beignet.polynomial._hermtrim.hermtrim(
                    beignet.polynomial._hermder.hermder(
                        beignet.polynomial._hermint.hermint(tgt, m=j, scl=2),
                        m=j,
                        scl=0.5,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial._hermtrim.hermtrim(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial._hermder.hermder(c) for c in c2d.T]).T
    numpy.testing.assert_almost_equal(
        beignet.polynomial._hermder.hermder(c2d, axis=0), tgt
    )

    tgt = numpy.vstack([beignet.polynomial._hermder.hermder(c) for c in c2d])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._hermder.hermder(c2d, axis=1), tgt
    )
