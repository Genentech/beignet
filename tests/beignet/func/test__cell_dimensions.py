import hypothesis.extra.numpy
import hypothesis.strategies as st
import math
import pytest
import torch
from torch import Tensor

from beignet.func._molecular_dynamics._partition.__cell_dimensions import (
    _cell_dimensions,
)


@st.composite
def _cell_dimensions_strategy(draw):
    spatial_dimension = draw(st.integers(min_value=1, max_value=10))

    box_size = draw(
        st.one_of(
            st.integers(min_value=3, max_value=10),
            st.floats(min_value=3.0, max_value=10.0),
            st.lists(
                st.floats(min_value=3.0, max_value=10.0),
                min_size=spatial_dimension,
                max_size=spatial_dimension,
            )
            .map(torch.tensor)
            .map(lambda x: x.float()),
        )
    )

    minimum_cell_size = draw(st.floats(min_value=1.0, max_value=10.0))

    return spatial_dimension, box_size, minimum_cell_size


@pytest.mark.parametrize(
    "spatial_dimension, box_size, minimum_cell_size, expected_exception",
    [
        (3, 100, 10, None),
        (3, 1, 10, ValueError),
        (2, torch.tensor([100, 100]), 10.0, None),
        (2, torch.tensor([1, 1]), 10.0, ValueError),
        (0, torch.tensor([100]), 10.0, AssertionError),
    ],
)
def test_cell_dimensions_exceptions(
    spatial_dimension, box_size, minimum_cell_size, expected_exception
):
    if expected_exception is not None:
        with pytest.raises(expected_exception):
            _cell_dimensions(spatial_dimension, box_size, minimum_cell_size)
    else:
        _cell_dimensions(spatial_dimension, box_size, minimum_cell_size)


@hypothesis.given(_cell_dimensions_strategy())
def test__cell_dimensions(data):
    spatial_dimension, box_size, minimum_cell_size = data

    if isinstance(box_size, (int, float)) and box_size < minimum_cell_size:
        with pytest.raises(ValueError):
            _cell_dimensions(spatial_dimension, box_size, minimum_cell_size)
        return

    if isinstance(box_size, Tensor):
        flattened_box_size = box_size.float().flatten()
        if any(size < minimum_cell_size * 3 for size in flattened_box_size):
            with pytest.raises(ValueError):
                _cell_dimensions(spatial_dimension, box_size, minimum_cell_size)
            return

    box_size_out, cell_size, cells_per_side, cell_count = _cell_dimensions(
        spatial_dimension, box_size, minimum_cell_size
    )

    if isinstance(box_size, (int, float)):
        assert box_size_out == float(box_size)

        assert math.isclose(float(box_size_out / cells_per_side), cell_size)

    elif isinstance(box_size, Tensor):
        assert torch.equal(box_size.float(), box_size_out)

        torch.testing.assert_allclose(box_size / cells_per_side.float(), cell_size)

    expected_cell_count = (
        int(torch.prod(cells_per_side).item())
        if isinstance(cells_per_side, Tensor)
        else int(cells_per_side**spatial_dimension)
    )

    assert cell_count == expected_cell_count
