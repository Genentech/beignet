import beignet.polynomial
import numpy


def test__trim_sequence():
    for num_trailing_zeros in range(5):
        numpy.testing.assert_equal(
            beignet.polynomial._trim_sequence([1] + [0] * num_trailing_zeros), [1]
        )

    for empty_seq in [[], numpy.array([], dtype=numpy.int32)]:
        numpy.testing.assert_equal(
            beignet.polynomial._trim_sequence(empty_seq), empty_seq
        )
