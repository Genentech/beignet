import hypothesis.strategies as st
import pytest
import torch
import numpy as np
from hypothesis import given

from beignet import iota


@st.composite
def _iota_strategy(draw):
    max_dimensions = 5
    dim = draw(st.integers(min_value=0, max_value=max_dimensions - 1))

    shape = tuple(
        draw(
            st.lists(
                st.integers(min_value=1, max_value=10),
                min_size=1,
                max_size=max_dimensions,
            )
        )
    )

    kwargs = {
        "dtype": draw(
            st.sampled_from([torch.int32, torch.int64, torch.float32, torch.float64])
        ),
        "device": draw(
            st.sampled_from(["cpu", "cuda"])
            if torch.cuda.is_available()
            else st.just("cpu")
        ),
    }

    return shape, dim, kwargs


@given(_iota_strategy())
def test_iota(data):
    shape, dim, kwargs = data

    # Check for dim out of range
    if dim >= len(shape):
        with pytest.raises(IndexError):
            iota(shape, dim, **kwargs)
        return

    result = iota(shape, dim, **kwargs)

    # Validate result shape
    assert result.shape == shape

    # Validate the content of the tensor along the specified dimension
    for idx in range(shape[dim]):
        if len(shape) > 1:
            assert torch.equal(
                result.select(dim, idx),
                torch.tensor(idx, **kwargs).expand(*result.select(dim, idx).shape),
            )
        else:
            assert result[idx].item() == idx

    # Compare with numpy equivalent
    np_result = np.indices(shape)[dim]
    np_result = torch.tensor(np_result, **kwargs)
    assert torch.equal(result, np_result)

    # Additional parametric tests
    parametric_tests = [
        ((3, 4), 5, IndexError),  # `dim` out of range
        ((3,), 0, None),  # Valid input
    ]

    for shape, dim, expected_exception in parametric_tests:
        if expected_exception is not None:
            with pytest.raises(expected_exception):
                iota(shape, dim)
        else:
            iota(shape, dim)