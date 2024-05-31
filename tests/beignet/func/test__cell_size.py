import torch
import hypothesis.strategies as st
import pytest
from hypothesis import given

from beignet.func._molecular_dynamics._partition.__cell_size import _cell_size


@st.composite
def _cell_size_strategy(draw):
    shape = draw(st.integers(min_value=1, max_value=10))

    box = torch.tensor(
        draw(
            st.lists(
                st.floats(min_value=1.0, max_value=100.0),
                min_size=shape,
                max_size=shape,
            )
        ),
        dtype=torch.float32,
    )

    minimum_unit_size = torch.tensor(
        draw(
            st.lists(
                st.floats(min_value=1.0, max_value=10.0), min_size=shape, max_size=shape
            )
        ),
        dtype=torch.float32,
    )

    return box, minimum_unit_size


@pytest.mark.parametrize(
    "box, minimum_unit_size, expected_exception",
    [
        (torch.tensor([10.0, 20.0]), torch.tensor([1.0, 2.0]), None),
        (torch.tensor([10.0, 20.0, 5.0]), torch.tensor([1.0, 2.0]), ValueError),
    ],
)
def test_cell_size_exceptions(box, minimum_unit_size, expected_exception):
    if expected_exception is not None:
        with pytest.raises(expected_exception):
            _cell_size(box, minimum_unit_size)
    else:
        _cell_size(box, minimum_unit_size)


@given(_cell_size_strategy())
def test__cell_size(data):
    box, minimum_unit_size = data

    # Skip zero values for minimum_unit_size to avoid dividing by zero.
    if (minimum_unit_size == 0).any():
        hypothesis.reject()

    result = _cell_size(box, minimum_unit_size)
    expected_result = box / torch.floor(box / minimum_unit_size)

    assert torch.allclose(result, expected_result)
