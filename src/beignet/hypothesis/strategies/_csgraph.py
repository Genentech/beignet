from typing import List

import hypothesis.strategies
import numpy
import scipy.sparse


@hypothesis.strategies.composite
def csr_array_graph(
    draw,
    *,
    dtypes: List[numpy.dtype] | None = None,
    integer_weights: bool = False,
    allow_negative: bool = True,
    force_int32_indices: bool = False,
) -> scipy.sparse.csr_array:
    """
    Generate CSR arrays representing graphs with configurable weight properties.

    Parameters
    ----------
    dtypes : List[numpy.dtype] | None
        List of allowed data types. If None, defaults based on integer_weights.
    integer_weights : bool
        If True, generate integer weights only (useful for flow algorithms).
        If False, generate floating-point weights.
    allow_negative : bool
        If True, allow negative weights. If False, generate only non-negative weights
        (useful for algorithms that don't handle negative cycles).
    force_int32_indices : bool
        If True, force indices and indptr arrays to be int32 dtype for SciPy
        compatibility.
    """
    n = draw(
        hypothesis.strategies.integers(
            min_value=1,
            max_value=200,
        ),
    )

    if dtypes is None:
        if integer_weights:
            dtypes = [numpy.int32, numpy.int64]
        else:
            dtypes = [numpy.float32, numpy.float64, numpy.int32, numpy.int64]

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

    # Generate weights based on configuration
    if integer_weights:
        if allow_negative:
            weight_strategy = hypothesis.strategies.integers(
                min_value=-100,
                max_value=100,
            )
        else:
            weight_strategy = hypothesis.strategies.integers(
                min_value=1,
                max_value=100,
            )
    else:
        if allow_negative:
            weight_strategy = hypothesis.strategies.floats(
                min_value=-10,
                max_value=+10,
            )
        else:
            weight_strategy = hypothesis.strategies.floats(
                min_value=0.1,
                max_value=+10,
            )

    csr = scipy.sparse.csr_array(
        (
            numpy.asarray(
                draw(
                    hypothesis.strategies.lists(
                        weight_strategy,
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

    # Force int32 indices for SciPy compatibility if requested
    if force_int32_indices:
        csr.indices = csr.indices.astype(numpy.int32)
        csr.indptr = csr.indptr.astype(numpy.int32)

    return csr


# Convenience functions for backward compatibility and common use cases
def csr_array_no_negative_cycles(**kwargs):
    """Generate CSR arrays without negative cycles (no negative weights)."""
    kwargs.setdefault("force_int32_indices", True)
    return csr_array_graph(allow_negative=False, **kwargs)


def csr_array_integer_weights(**kwargs):
    """Generate CSR arrays with integer weights for flow algorithms."""
    kwargs.setdefault("force_int32_indices", True)
    return csr_array_graph(integer_weights=True, allow_negative=False, **kwargs)
