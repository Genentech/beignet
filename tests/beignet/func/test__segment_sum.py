import hypothesis.strategies as st
import math
import pytest
import torch
from hypothesis import given

from beignet.func._molecular_dynamics._partition.__segment_sum import _segment_sum


@st.composite
def _segment_sum_strategy(draw):
    num_elements = draw(st.integers(min_value=1, max_value=10))

    inner_shape = draw(
        st.shared(
            st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=4)
        )
    )

    input = draw(
        st.lists(
            elements=st.floats(min_value=0.0, max_value=10.0),
            min_size=math.prod(inner_shape) * num_elements,
            max_size=math.prod(inner_shape) * num_elements,
        )
        .map(torch.tensor)
        .map(lambda x: x.view(num_elements, *inner_shape))
    )

    indexes = draw(
        st.lists(
            elements=st.integers(min_value=0, max_value=num_elements - 1),
            min_size=num_elements,
            max_size=num_elements,
        ).map(torch.tensor)
    )

    n = draw(
        st.one_of(
            st.none(),
            st.integers(min_value=indexes.max().item() + 1, max_value=num_elements * 2),
        )
    )

    return input, indexes, n


@pytest.mark.parametrize(
    "input, indexes, n, expected_exception",
    [
        (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([0, 1, 2]),
            None,
            ValueError,
        ),  # Indexes not matching length of input
        (torch.tensor([1.0, 2.0]), torch.tensor([0, 1]), 3.0, TypeError),
        # Invalid Type for n
    ],
)
def test_segment_sum_exceptions(input, indexes, n, expected_exception):
    """
    Test the `_segment_sum` function for expected exceptions based on input parameters.
    """
    if expected_exception is not None:
        with pytest.raises(expected_exception):
            _segment_sum(input, indexes, n)

    else:
        _segment_sum(input, indexes, n)


@given(_segment_sum_strategy())
def test__segment_sum(data):
    """
    Property-based test for the `_segment_sum` function ensuring correct behavior for various inputs.
    """
    input, indexes, n = data

    # Check for invalid dimensions
    if indexes.ndim != 1 or indexes.shape[0] != input.shape[0]:
        with pytest.raises(ValueError):
            _segment_sum(input, indexes, n)

        return

    result = _segment_sum(input, indexes, n)

    # Validate the result shape
    expected_shape = (n if n is not None else (max(indexes) + 1), *input.shape[1:])

    assert result.shape == expected_shape

    # Validate the sum of the result tensor
    for i in range(result.shape[0]):
        segmented_sum = input[indexes == i].sum(dim=0)

        assert torch.allclose(result[i], segmented_sum)
