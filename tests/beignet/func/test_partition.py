import unittest
from functools import partial
from typing import Callable
from unittest.mock import patch

import hypothesis.extra.numpy
import hypothesis.strategies as st
import math
import pytest
import torch
from hypothesis import given
from torch import Tensor, vmap
from torch.testing import assert_allclose

from beignet import map_bond, map_product
from beignet.func import space
from beignet.func._partition import (
    _cell_dimensions,
    _cell_size,
    _hash_constants,
    _normalize_cell_size,
    _particles_per_cell,
    segment_sum,
    _shift,
    _to_square_metric_fn,
    _unflatten_cell_buffer,
    cell_list,
    metric,
    neighbor_list,
    safe_index,
    _NeighborListFormat,
    _neighbor_list_mask,
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


def test_hash_constants_uniform_grid():
    spatial_dimensions = 3
    cells_per_side = torch.tensor([2])
    expected_output = torch.tensor([[1, 2, 4]], dtype=torch.int32)
    assert torch.equal(
        _hash_constants(spatial_dimensions, cells_per_side), expected_output
    )


def test_hash_constants_invalid_input_size():
    spatial_dimensions = 3
    cells_per_side = torch.tensor([2, 3])
    with pytest.raises(ValueError):
        _hash_constants(spatial_dimensions, cells_per_side)


def test_hash_constants_zero_dimensions():
    spatial_dimensions = 3
    cells_per_side = torch.tensor([])
    with pytest.raises(ValueError):
        _hash_constants(spatial_dimensions, cells_per_side)


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


@patch("beignet.func._partition._cell_dimensions")
@patch("beignet.func._partition._hash_constants")
@patch("beignet.func._partition.segment_sum")
def test_particles_per_cell(
    mock_segment_sum, mock_hash_constants, mock_cell_dimensions
):
    positions = torch.tensor([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
    size = torch.tensor([3.0, 3.0])
    minimum_size = 1.0

    mock_cell_dimensions.return_value = (
        size,
        torch.tensor([1.0, 1.0]),
        torch.tensor([3, 3]),
        9,
    )
    mock_hash_constants.return_value = torch.tensor([1, 3], dtype=torch.int32)
    mock_segment_sum.return_value = torch.tensor(
        [1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=torch.int32
    )

    expected_output = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=torch.int32)

    output = _particles_per_cell(positions, size, minimum_size)

    assert torch.equal(output, expected_output)
    mock_cell_dimensions.assert_called_once_with(2, size, minimum_size)

    assert mock_hash_constants.call_count == 1
    args, _ = mock_hash_constants.call_args
    assert args[0] == 2
    assert torch.equal(args[1], torch.tensor([3, 3]))

    mock_segment_sum.assert_called_once()


def test_segment_sum():
    one_particle_hash = torch.tensor([[1]])
    particle_hash = torch.tensor([[6]])
    cell_count = 4

    assert torch.equal(
        segment_sum(one_particle_hash, particle_hash, cell_count),
        torch.tensor([[0], [0], [0], [0]]),
    )


@st.composite
def _shift_strategy(draw):
    max_dims = draw(st.integers(min_value=2, max_value=3))
    shape = draw(
        st.lists(
            st.integers(min_value=2, max_value=5), min_size=max_dims, max_size=max_dims
        )
    )

    tensor = draw(
        st.lists(
            st.floats(min_value=-5.0, max_value=5.0),
            min_size=torch.prod(torch.tensor(shape)).item(),
            max_size=torch.prod(torch.tensor(shape)).item(),
        )
        .map(torch.tensor)
        .map(lambda x: x.view(*shape))
    )

    shift_vec = draw(
        st.lists(
            st.integers(min_value=-shape[0], max_value=shape[0]),
            min_size=max_dims,
            max_size=max_dims,
        ).map(torch.tensor)
    )

    return tensor, shift_vec


@given(_shift_strategy())
def test__shift(data):
    """
    Property-based test for the `_shift` function ensuring correct behavior for various inputs.
    """
    a, b = data

    result = _shift(a, b)

    assert result.shape == a.shape

    expected = torch.roll(a, shifts=tuple(b.numpy()), dims=tuple(range(len(b))))
    assert torch.equal(result, expected)


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


def test__unflatten_cell_buffer_cells_per_side_is_scalar():
    buffer = torch.arange(60).reshape(10, 2, 3)
    cells_per_side = torch.tensor(5)
    dim = 1

    assert torch.equal(
        _unflatten_cell_buffer(buffer, cells_per_side, dim),
        torch.arange(60).reshape(5, 2, 2, 3),
    )


def test__unflatten_cell_buffer_cells_per_side_is_1d_tensor():
    buffer = torch.arange(60).reshape(10, 2, 3)
    cells_per_side = torch.tensor([5])
    dim = 1

    expected_result = torch.arange(60).reshape(5, 2, 2, 3)

    result = _unflatten_cell_buffer(buffer, cells_per_side, dim)

    assert torch.equal(
        result, expected_result
    ), f"Expected: {expected_result}, but got: {result}"


@pytest.fixture
def expected_tensor():
    return torch.tensor(
        [
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            [[8, 9], [10, 11], [12, 13], [14, 15]],
            [[16, 17], [18, 19], [20, 21], [22, 23]],
        ]
    )


def test__unflatten_cell_buffer_cells_per_side_is_1d_tensor_length_2(expected_tensor):
    buffer = torch.arange(24)
    cells_per_side = torch.tensor([4, 3])
    dim = 2

    expected_shape = (3, 4, 2)

    result = _unflatten_cell_buffer(buffer, cells_per_side, dim)

    assert torch.equal(
        result, expected_tensor
    ), f"Expected: {expected_tensor}, but got: {result}"
    assert result.shape == expected_shape


def test__unflatten_cell_buffer_cells_per_side_2d_length_2(expected_tensor):
    buffer = torch.arange(24)
    cells_per_side = torch.tensor([[4, 3]])
    dim = 2

    expected_shape = (3, 4, 2)

    result = _unflatten_cell_buffer(buffer, cells_per_side, dim)

    assert torch.equal(
        result, expected_tensor
    ), f"Expected: {expected_tensor}, but got: {result}"
    assert result.shape == expected_shape


def test__unflatten_cell_buffer_cells_per_side_2d_with_values_in_each_dimension(
    expected_tensor,
):
    buffer = torch.arange(24)
    cells_per_side = torch.tensor([[4, 3], [2, 3]])
    dim = 2

    expected_shape = (3, 4, 2)

    result = _unflatten_cell_buffer(buffer, cells_per_side, dim)

    assert torch.equal(
        result, expected_tensor
    ), f"Expected: {expected_tensor}, but got: {result}"
    assert result.shape == expected_shape


class CellListTest(unittest.TestCase):
    def test_cell_list_emplace_2d(self):
        dtype = torch.float32
        box_size = torch.tensor([8.65, 8.0], dtype=torch.float32)
        cell_size = 1.0

        # Test particle positions
        R = torch.tensor(
            [[0.25, 0.25], [8.5, 1.95], [8.1, 1.5], [3.7, 7.9]], dtype=dtype
        )

        cell_fn = cell_list(box_size, cell_size)

        cell_list_instance = cell_fn.setup_fn(R)

        self.assertEqual(cell_list_instance.indices.dtype, torch.int32)

        assert_allclose(R[0], cell_list_instance.positions_buffer[0, 0, 0])
        assert_allclose(R[1], cell_list_instance.positions_buffer[1, 7, 1])
        assert_allclose(R[2], cell_list_instance.positions_buffer[1, 7, 0])
        assert_allclose(R[3], cell_list_instance.positions_buffer[7, 3, 1])

        self.assertEqual(0, cell_list_instance.indices[0, 0, 0])
        self.assertEqual(1, cell_list_instance.indices[1, 7, 1])
        self.assertEqual(2, cell_list_instance.indices[1, 7, 0])
        self.assertEqual(3, cell_list_instance.indices[7, 3, 1])

        id_flat = cell_list_instance.indices.view(-1)
        R_flat = cell_list_instance.positions_buffer.view(-1, 2)

        R_out = torch.zeros((5, 2), dtype=dtype)
        R_out[id_flat] = R_flat
        R_out = R_out[:-1]

        assert_allclose(R_out, R)

    def test_cell_list_random_emplace(self):
        dtype = torch.float32
        dim = 2  # Change this to 3 if you want to test for 3D
        particle_count = 10
        torch.manual_seed(1)

        box_size = 9.0
        cell_size = 1.0

        R = box_size * torch.rand((particle_count, dim), dtype=dtype)

        cell_fn = cell_list(torch.tensor([box_size] * dim, dtype=dtype), cell_size)
        cell_list_instance = cell_fn.setup_fn(R)

        id_flat = cell_list_instance.indices.view(-1)
        R_flat = cell_list_instance.positions_buffer.view(-1, dim)
        R_out = torch.zeros((particle_count + 1, dim), dtype=dtype)
        R_out[id_flat] = R_flat
        R_out = R_out[:-1]

        assert_allclose(R_out, R)

    def test_cell_list_random_emplace_rect(self):
        dtype = torch.float32
        dim = 2  # Change this to 3 if you want to test for 3D
        particle_count = 10
        torch.manual_seed(1)

        box_size = (
            torch.tensor([9.0, 3.25], dtype=dtype)
            if dim == 2
            else torch.tensor([9.0, 3.0, 7.25], dtype=dtype)
        )
        cell_size = 1.0

        R = box_size * torch.rand((particle_count, dim), dtype=dtype)

        cell_fn = cell_list(box_size, cell_size)
        cell_list_instance = cell_fn.setup_fn(R)

        id_flat = cell_list_instance.indices.view(-1)
        R_flat = cell_list_instance.positions_buffer.view(-1, dim)
        R_out = torch.zeros((particle_count + 1, dim), dtype=dtype)
        R_out[id_flat] = R_flat
        R_out = R_out[:-1]

        assert_allclose(R_out, R)

    def test_cell_list_random_emplace_side_data(self):
        dtype = torch.float32
        dim = 2  # Change this to 3 if you want to test for 3D
        particle_count = 10
        torch.manual_seed(1)

        box_size = (
            torch.tensor([9.0, 4.25], dtype=dtype)
            if dim == 2
            else torch.tensor([9.0, 4.0, 7.25], dtype=dtype)
        )
        cell_size = 1.23

        R = box_size * torch.rand((particle_count, dim), dtype=dtype)
        side_data_dim = 2
        side_data = torch.randn((particle_count, side_data_dim), dtype=dtype)

        cell_fn = cell_list(box_size, cell_size)
        cell_list_instance = cell_fn.setup_fn(R, side_data=side_data)

        id_flat = cell_list_instance.indices.view(-1)
        R_flat = cell_list_instance.positions_buffer.view(-1, dim)
        R_out = torch.zeros((particle_count + 1, dim), dtype=dtype)
        R_out[id_flat] = R_flat
        R_out = R_out[:-1]

        side_data_flat = cell_list_instance.parameters["side_data"].view(
            -1, side_data_dim
        )
        side_data_out = torch.zeros((particle_count + 1, side_data_dim), dtype=dtype)
        side_data_out[id_flat] = side_data_flat
        side_data_out = side_data_out[:-1]

        assert_allclose(R_out, R)
        assert_allclose(side_data_out, side_data)


PARTICLE_COUNT = 1000
POSITION_DTYPE = [torch.float32, torch.float64]  # Example values
SPATIAL_DIMENSION = [2, 3]
test_cases = [
    {
        "dtype": dtype,
        "dim": dim,
    }
    for dtype in POSITION_DTYPE
    for dim in SPATIAL_DIMENSION
]

# Extract the parameters and ids
params = [(case["dtype"], case["dim"]) for case in test_cases]


@pytest.mark.parametrize("dtype, dim", params)
def test_neighbor_list_build(dtype, dim):
    torch.manual_seed(1)

    box_size = (
        torch.tensor([9.0, 4.0, 7.25], dtype=torch.float32)
        if dim == 3
        else torch.tensor([9.0, 4.25], dtype=torch.float32)
    )
    cutoff = torch.tensor(1.23, dtype=torch.float32)

    displacement, _ = space(box=box_size, parallelepiped=False)

    metric_fn = metric(displacement)

    R = box_size * torch.rand((PARTICLE_COUNT, dim), dtype=dtype)
    N = R.shape[0]

    neighbor_fn = neighbor_list(displacement, box_size, cutoff, 0.0, 1.1)

    idx = neighbor_fn.setup_fn(R).indices

    R_neigh = safe_index(R, idx)

    mask = idx < N

    d = vmap(vmap(metric_fn, in_dims=(None, 0)))

    dR = d(R, R_neigh)

    d_exact = map_product(metric_fn)
    dR_exact = d_exact(R, R)

    dR = torch.where(dR < cutoff, dR, torch.tensor(0, dtype=torch.float32)) * mask

    mask_exact = 1.0 - torch.eye(dR_exact.shape[0])
    dR_exact = (
        torch.where(dR_exact < cutoff, dR_exact, torch.tensor(0, dtype=torch.float32))
        * mask_exact
    )

    dR, _ = torch.sort(dR, dim=1)
    dR_exact, _ = torch.sort(dR_exact, dim=1)

    for i in range(dR.shape[0]):
        dR_row = dR[i]
        dR_row = dR_row[dR_row > 0.0]

        dR_exact_row = dR_exact[i]
        dR_exact_row = torch.tensor(dR_exact_row[dR_exact_row > 0.0], dtype=dtype)

        assert torch.allclose(dR_row, dR_exact_row)


@pytest.mark.parametrize("dtype, dim", params)
def test_neighbor_list_build_sparse(dtype, dim):
    torch.manual_seed(1)

    box_size = (
        torch.tensor([9.0, 4.0, 7.25], dtype=torch.float32)
        if dim == 3
        else torch.tensor([9.0, 4.25], dtype=torch.float32)
    )
    cutoff = torch.tensor(1.23, dtype=torch.float32)

    displacement, _ = space(box=box_size, parallelepiped=False)
    metric_fn = metric(displacement)

    R = box_size * torch.rand((16, dim), dtype=dtype)
    N = R.shape[0]

    neighbor_fn = neighbor_list(
        displacement,
        box_size,
        cutoff,
        0.0,
        1.1,
        neighbor_list_format=_NeighborListFormat.SPARSE,
    )

    nbrs = neighbor_fn.setup_fn(R)
    mask = _neighbor_list_mask(nbrs)

    d = map_bond(metric_fn)
    dR = d(safe_index(R, nbrs.indices[0]), safe_index(R, nbrs.indices[1]))

    d_exact = map_product(metric_fn)
    dR_exact = d_exact(R, R)

    dR = torch.where(dR < cutoff, dR, torch.tensor(0)) * mask
    mask_exact = 1.0 - torch.eye(dR_exact.shape[0])
    dR_exact = torch.where(dR_exact < cutoff, dR_exact, torch.tensor(0)) * mask_exact

    dR_exact, _ = torch.sort(dR_exact, dim=1)

    for i in range(N):
        dR_row = dR[nbrs.indices[0] == i]
        dR_row = dR_row[dR_row > 0.0]
        dR_row, _ = torch.sort(dR_row)

        dR_exact_row = dR_exact[i]
        dR_exact_row = torch.tensor(dR_exact_row[dR_exact_row > 0.0], dtype=dtype)

        assert torch.allclose(dR_row, dR_exact_row)


def test_cell_list_overflow():
    displacement_fn, shift_fn = space()

    box = torch.tensor(100.0)
    r_cutoff = 3.0
    dr_threshold = 0.0

    neighbor_fn = neighbor_list(
        distance_fn=displacement_fn,
        box=box,
        neighborhood_radius=r_cutoff,
        maximum_distance=dr_threshold,
    )

    # all far from eachother
    positions = torch.tensor(
        [
            [20.0, 20.0],
            [30.0, 30.0],
            [40.0, 40.0],
            [50.0, 50.0],
        ]
    )

    neighbors = neighbor_fn.setup_fn(positions)

    assert neighbors.indices.dtype is torch.int32

    # two first point are close to eachother
    positions = torch.tensor(
        [
            [20.0, 20.0],
            [20.0, 20.0],
            [40.0, 40.0],
            [50.0, 50.0],
        ]
    )

    neighbors = neighbor_fn.update_fn(positions, neighbors)

    assert neighbors.did_buffer_overflow
    assert neighbors.indices.dtype is torch.int32


def test_custom_mask_function():
    displacement_fn, shift_fn = space()

    box = torch.tensor(1.0)
    r_cutoff = 3.0
    dr_threshold = 0.0
    n_particles = 10
    R = torch.zeros(3).expand(n_particles, 3)

    def acceptable_id_pair(id1, id2):
        """
        Don't allow particles to have an interaction when their id's
        are closer than 3 (eg disabling 1-2 and 1-3 interactions)
        """
        return torch.abs(id1 - id2) > 3

    def mask_id_based(
        idx: Tensor, ids: Tensor, mask_val: int, _acceptable_id_pair: Callable
    ) -> Tensor:
        """
        _acceptable_id_pair mapped to act upon the neighbor list where:
        - index of particle 1 is in index in the first dimension of array
        - index of particle 2 is given by the value in the array
        """

        @partial(vmap, in_dims=(0, 0, None))
        def acceptable_id_pair(idx, id1, ids):
            id2 = safe_index(ids, idx)

            return vmap(_acceptable_id_pair, in_dims=(None, 0))(id1, id2)

        mask = acceptable_id_pair(idx, ids, ids)

        return torch.where(mask, idx, mask_val)

    ids = torch.arange(n_particles)  # id is just particle index here.
    mask_val = n_particles
    custom_mask_function = partial(
        mask_id_based,
        ids=ids,
        mask_val=mask_val,
        _acceptable_id_pair=acceptable_id_pair,
    )

    neighbor_fn = neighbor_list(
        distance_fn=displacement_fn,
        box=box,
        neighborhood_radius=r_cutoff,
        maximum_distance=dr_threshold,
        mask_fn=custom_mask_function,
    )

    neighbors = neighbor_fn.setup_fn(R)
    neighbors = neighbors.update_fn(R)
    """
    Without masking it's 9 neighbors (with mask self) -> 90 neighbors.
    With masking -> 42.
    """
    assert 42 == (neighbors.indices != mask_val).sum()


def test_issue191_1():
    box_vector = torch.ones(3) * 3

    r_cut = 0.1
    _positions = torch.linspace(0.5, 0.7, 20)
    positions = torch.stack([_positions, _positions, _positions], dim=1)

    displacement, _ = space(box_vector, parallelepiped=True)

    neighbor_fn = neighbor_list(
        displacement,
        box_vector,
        r_cut,
        0.1 * r_cut,
        normalized=True,
    )

    neighbor2_fn = neighbor_list(
        displacement,
        box_vector[0],
        r_cut,
        0.1 * r_cut,
        normalized=True,
        disable_unit_list=True,
    )

    nbrs = neighbor_fn.setup_fn(positions)
    nbrs2 = neighbor2_fn.setup_fn(positions)

    tensor_1, _ = torch.sort(nbrs.indices, dim=-1)
    tensor_2, _ = torch.sort(nbrs2.indices, dim=-1)

    assert torch.allclose(tensor_1, tensor_2)


@pytest.mark.parametrize(
    "r_cut, disable_cell_list, capacity_multiplier, mask_self, fmt",
    [
        (0.12, True, 1.5, False, _NeighborListFormat.DENSE),
        (0.12, True, 1.5, False, _NeighborListFormat.SPARSE),
        (0.12, True, 1.5, False, _NeighborListFormat.ORDERED_SPARSE),
        (0.12, True, 1.5, True, _NeighborListFormat.DENSE),
        (0.12, True, 1.5, True, _NeighborListFormat.SPARSE),
        (0.12, True, 1.5, True, _NeighborListFormat.ORDERED_SPARSE),
        (0.25, False, 1.5, False, _NeighborListFormat.DENSE),
        (0.25, False, 1.5, False, _NeighborListFormat.SPARSE),
        (0.25, False, 1.5, False, _NeighborListFormat.ORDERED_SPARSE),
        (0.25, False, 1.5, True, _NeighborListFormat.DENSE),
        (0.25, False, 1.5, True, _NeighborListFormat.SPARSE),
        (0.25, False, 1.5, True, _NeighborListFormat.ORDERED_SPARSE),
        (0.31, False, 1.5, False, _NeighborListFormat.DENSE),
        (0.31, False, 1.5, False, _NeighborListFormat.SPARSE),
        (0.31, False, 1.5, False, _NeighborListFormat.ORDERED_SPARSE),
        (0.31, False, 1.5, True, _NeighborListFormat.DENSE),
        (0.31, False, 1.5, True, _NeighborListFormat.SPARSE),
        (0.31, False, 1.5, True, _NeighborListFormat.ORDERED_SPARSE),
        (0.31, False, 1.0, False, _NeighborListFormat.DENSE),
        (0.31, False, 1.0, False, _NeighborListFormat.SPARSE),
        (0.31, False, 1.0, False, _NeighborListFormat.ORDERED_SPARSE),
        (0.31, False, 1.0, True, _NeighborListFormat.DENSE),
        (0.31, False, 1.0, True, _NeighborListFormat.SPARSE),
        (0.31, False, 1.0, True, _NeighborListFormat.ORDERED_SPARSE),
    ],
)
def test_issue191_2(r_cut, disable_cell_list, capacity_multiplier, mask_self, fmt):
    box = torch.ones(3)
    # box = 1.0
    if fmt is _NeighborListFormat.DENSE:
        desired_shape = (20, 19) if mask_self else (20, 20)

        _positions = torch.ones((20,)) * 0.5

    elif fmt is _NeighborListFormat.SPARSE:
        desired_shape = (2, 20 * 19) if mask_self else (2, 20**2)

        _positions = torch.ones((20,)) * 0.5

    elif fmt is _NeighborListFormat.ORDERED_SPARSE:
        desired_shape = (2, 20 * 19 // 2)

        _positions = torch.ones((20,)) * 0.5

    positions = torch.stack([_positions, _positions, _positions], dim=1)

    displacement, _ = space(box=box, parallelepiped=False)

    neighbor_fn = neighbor_list(
        displacement,
        box,
        r_cut,
        0.1 * r_cut,
        buffer_size_multiplier=capacity_multiplier,
        disable_unit_list=disable_cell_list,
        mask_self=mask_self,
        neighbor_list_format=fmt,
    )

    nbrs = neighbor_fn.setup_fn(positions)

    assert nbrs.did_buffer_overflow is False
    assert nbrs.indices.shape == desired_shape

    new_nbrs = neighbor_fn.update_fn(positions + 0.1, nbrs)

    assert new_nbrs.did_buffer_overflow is False
    assert new_nbrs.indices.shape == desired_shape
