from typing import List

import hypothesis.strategies
import numpy
import scipy.sparse


@hypothesis.strategies.composite
def csr_array(
    draw,
    *,
    dtypes: List[numpy.dtype] | None = None,
) -> scipy.sparse.csr_array:
    n = draw(
        hypothesis.strategies.integers(
            min_value=1,
            max_value=200,
        ),
    )

    if dtypes is None:
        dtypes = [
            numpy.float32,
            numpy.float64,
            numpy.int32,
            numpy.int64,
        ]

    density = draw(
        hypothesis.strategies.floats(
            min_value=0.0,
            max_value=0.15,
        ),
    )

    min_size = draw(
        hypothesis.strategies.integers(
            min_value=1,
            max_value=max(1, int(density * n * n)),
        ),
    )

    return scipy.sparse.csr_array(
        (
            numpy.asarray(
                draw(
                    hypothesis.strategies.lists(
                        hypothesis.strategies.floats(
                            min_value=-10,
                            max_value=+10,
                        ),
                        min_size=min_size,
                        max_size=min_size,
                    ),
                ),
                dtype=draw(
                    hypothesis.strategies.sampled_from(
                        dtypes,
                    ),
                ),
            ),
            (
                numpy.asarray(
                    draw(
                        hypothesis.strategies.lists(
                            hypothesis.strategies.integers(0, n - 1),
                            min_size=min_size,
                            max_size=min_size,
                        )
                    )
                ),
                numpy.asarray(
                    draw(
                        hypothesis.strategies.lists(
                            hypothesis.strategies.integers(0, n - 1),
                            min_size=min_size,
                            max_size=min_size,
                        )
                    ),
                ),
            ),
        ),
        shape=(n, n),
    )
