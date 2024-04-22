from typing import Generator

import numpy


def _neighboring_cell_lists(
    dimension: int,
) -> Generator[numpy.ndarray, None, None]:
    for index in numpy.ndindex(*([3] * dimension)):
        yield numpy.array(index, dtype=numpy.int32) - 1
