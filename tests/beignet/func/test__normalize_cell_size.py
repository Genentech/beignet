import hypothesis.strategies as st
import torch
import pytest
from hypothesis import given

from beignet.func._molecular_dynamics._partition.__normalize_cell_size import (
    _normalize_cell_size,
)


@st.composite
def _normalize_cell_size_strategy(draw):
    box_dim = draw(st.integers(min_value=0, max_value=2))

    if box_dim == 0:
        box = torch.tensor([draw(st.floats(min_value=1.0, max_value=10.0))])

    elif box_dim == 1:
        size = draw(st.integers(min_value=1, max_value=10))
        box = torch.tensor(
            draw(
                st.lists(
                    st.floats(min_value=1.0, max_value=10.0),
                    min_size=size,
                    max_size=size,
                )
            )
        )

    elif box_dim == 2:
        shape = draw(st.sampled_from([(1, 1), (2, 2), (3, 3)]))
        box = torch.tensor(
            draw(
                st.lists(
                    st.lists(
                        st.floats(min_value=1.0, max_value=10.0),
                        min_size=shape[1],
                        max_size=shape[1],
                    ),
                    min_size=shape[0],
                    max_size=shape[0],
                )
            )
        )

    cutoff = draw(st.floats(min_value=1.0, max_value=10.0))

    return box, cutoff


@pytest.mark.parametrize(
    "box, cutoff, expected_exception",
    [
        (
            torch.tensor([[10, 20, 30], [10, 20, 30], [10, 20, 30], [10, 20, 30]]),
            2.0,
            ValueError,
        ),
        (torch.tensor([[[[[10, 5], [3, 4], [1, 2]]]]]), 2.0, ValueError),
    ],
)
def test_normalize_cell_size_exceptions(box, cutoff, expected_exception):
    """
    Test the `_normalize_cell_size` function for expected exceptions based on input parameters.
    """
    if expected_exception is not None:
        with pytest.raises(expected_exception):
            _normalize_cell_size(box, cutoff)
    else:
        _normalize_cell_size(box, cutoff)


@given(_normalize_cell_size_strategy())
def test__normalize_cell_size(data):
    """
    Property-based test for the `_normalize_cell_size` function ensuring correct behavior for various inputs.
    """
    box, cutoff = data

    # Check for invalid shapes
    if box.ndim == 2 and box.shape[0] not in {1, 2, 3}:
        with pytest.raises(ValueError):
            _normalize_cell_size(box, cutoff)

        return

    result = _normalize_cell_size(box, cutoff)

    # Validate result
    if box.ndim == 0:
        expected_result = cutoff / box

    elif box.ndim == 1:
        expected_result = cutoff / torch.min(box)

    else:  # box.ndim == 2
        if box.shape[0] == 1:
            expected_result = 1 / torch.floor(box[0, 0] / cutoff)

        elif box.shape[0] == 2:
            xx = box[0, 0]
            yy = box[1, 1]
            xy = box[0, 1] / yy
            nx = xx / torch.sqrt(1 + xy**2)
            ny = yy
            nmin = torch.floor(torch.min(torch.tensor([nx, ny])) / cutoff)
            expected_result = 1 / torch.where(nmin == 0, 1, nmin)

        elif box.shape[0] == 3:
            xx = box[0, 0]
            yy = box[1, 1]
            zz = box[2, 2]
            xy = box[0, 1] / yy
            xz = box[0, 2] / zz
            yz = box[1, 2] / zz
            nx = xx / torch.sqrt(1 + xy**2 + (xy * yz - xz) ** 2)
            ny = yy / torch.sqrt(1 + yz**2)
            nz = zz
            nmin = torch.floor(torch.min(torch.tensor([nx, ny, nz])) / cutoff)
            expected_result = 1 / torch.where(nmin == 0, 1, nmin)

    assert torch.allclose(result, expected_result)
