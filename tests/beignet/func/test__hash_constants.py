import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given

from beignet.func._molecular_dynamics._partition.__hash_constants import _hash_constants


@st.composite
def _hash_constants_strategy(draw):
    spatial_dimensions = draw(st.integers(min_value=1, max_value=10))

    cells_per_side = draw(
        st.one_of(
            st.integers(min_value=1, max_value=10).map(
                lambda x: torch.tensor([x], dtype=torch.int32)
            ),
            st.lists(
                st.integers(min_value=1, max_value=10),
                min_size=spatial_dimensions,
                max_size=spatial_dimensions,
            ).map(lambda x: torch.tensor(x, dtype=torch.int32)),
        )
    )

    return spatial_dimensions, cells_per_side


@pytest.mark.parametrize(
    "spatial_dimensions, cells_per_side, expected_result, expected_exception",
    [
        (
            3,
            torch.tensor([4], dtype=torch.int32),
            torch.tensor([[1, 4, 16]], dtype=torch.int32),
            None,
        ),
        (
            3,
            torch.tensor([4, 4, 4], dtype=torch.int32),
            torch.tensor([1, 4, 16], dtype=torch.int32),
            None,
        ),
        (3, torch.tensor([4, 4], dtype=torch.int32), None, ValueError),
    ],
)
def test_hash_constants(
    spatial_dimensions, cells_per_side, expected_result, expected_exception
):
    if expected_exception is not None:
        with pytest.raises(expected_exception):
            _hash_constants(spatial_dimensions, cells_per_side)

    else:
        result = _hash_constants(spatial_dimensions, cells_per_side)
        assert torch.equal(result, expected_result)


@given(_hash_constants_strategy())
def test__hash_constants(data):
    spatial_dimensions, cells_per_side = data

    if cells_per_side.numel() == 1:
        expected_result = torch.tensor(
            [[cells_per_side.item() ** i for i in range(spatial_dimensions)]],
            dtype=torch.int32,
        )
    else:
        if cells_per_side.numel() != spatial_dimensions:
            with pytest.raises(ValueError):
                _hash_constants(spatial_dimensions, cells_per_side)

            return

        augmented = torch.cat(
            (
                torch.tensor([1], dtype=torch.int32).view(1, 1),
                cells_per_side[:-1].view(1, -1),
            ),
            dim=1,
        )

        expected_result = torch.cumprod(augmented.flatten(), dim=0)

    result = _hash_constants(spatial_dimensions, cells_per_side)

    assert torch.equal(result, expected_result)
