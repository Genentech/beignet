import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given
from torch import Tensor

from beignet.func._partition import _to_square_metric_fn


def dummy_fn(a: Tensor, b: Tensor, **kwargs) -> Tensor:
    return torch.norm(a - b, dim=-1)


@st.composite
def _square_metric_fn_strategy(draw):
    dimension = draw(st.integers(min_value=1, max_value=3))
    size = draw(st.integers(min_value=1, max_value=5))

    a = draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0),
            min_size=dimension,
            max_size=dimension,
        ).map(torch.tensor)
    )

    b = draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0),
            min_size=dimension,
            max_size=dimension,
        ).map(torch.tensor)
    )

    return a, b


@pytest.mark.parametrize(
    "a, b, expected_exception",
    [
        (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), None),
        # Valid input
        (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]), RuntimeError),
        # Mismatched dimensions
    ],
)
def test_square_metric_exceptions(a, b, expected_exception):
    """
    Test the `_to_square_metric_fn` function for expected exceptions based on input parameters.
    """
    square_metric = _to_square_metric_fn(dummy_fn)

    if expected_exception is not None:
        with pytest.raises(expected_exception):
            square_metric(a, b)
    else:
        square_metric(a, b)


@given(_square_metric_fn_strategy())
def test__square_metric_fn(data):
    """
    Property-based test for the `_to_square_metric_fn` function ensuring correct behavior for various inputs.
    """
    a, b = data

    square_metric = _to_square_metric_fn(dummy_fn)

    result = square_metric(a, b)

    # Validate the squared distance metric
    expected = torch.sum(torch.square(a - b), dim=-1)

    assert torch.allclose(result, expected)
