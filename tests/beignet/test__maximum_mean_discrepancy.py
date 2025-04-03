from typing import NamedTuple, Union

import numpy
import pytest
import torch
from beignet import maximum_mean_discrepancy
from numpy.testing import assert_allclose

ArrayType = Union[numpy.ndarray, torch.Tensor]


class TestData(NamedTuple):
    """Container for test arrays to ensure consistent initialization."""

    X: ArrayType
    Y: ArrayType
    X_same: ArrayType
    X_large: ArrayType
    Y_small: ArrayType


@pytest.fixture(scope="module")
def numpy_arrays() -> TestData:
    """Generate test arrays once per module using vectorized operations."""
    rng = numpy.random.default_rng(42)
    # Preallocate all arrays in one block
    X = rng.normal(0, 1, (1024, 2))
    Y = rng.normal(5, 1, (1024, 2))  # Different mean for clear separation
    X_large = rng.normal(0, 1, (150, 2))
    Y_small = rng.normal(0, 1, (50, 2))

    return TestData(
        X=X,
        Y=Y,
        X_same=X.copy(),  # Explicit copy for identity tests
        X_large=X_large,
        Y_small=Y_small,
    )


@pytest.fixture(scope="module")
def torch_arrays(numpy_arrays: TestData) -> TestData:
    """Convert numpy arrays to torch tensors, ensuring numpy is initialized first."""
    return TestData(
        X=torch.tensor(numpy_arrays.X),
        Y=torch.tensor(numpy_arrays.Y),
        X_same=torch.tensor(numpy_arrays.X_same),
        X_large=torch.tensor(numpy_arrays.X_large),
        Y_small=torch.tensor(numpy_arrays.Y_small),
    )


def manhattan_distance(x: ArrayType, y: ArrayType) -> ArrayType:
    """Vectorized Manhattan distance using array API operations."""
    xp = x.__array_namespace__()
    diff = xp.subtract(xp.expand_dims(x, 1), xp.expand_dims(y, 0))
    return xp.sum(xp.abs(diff), axis=-1)


@pytest.mark.parametrize("arrays", ["numpy_arrays", "torch_arrays"])
def test_mmd_basic(request, arrays):
    """Test core MMD functionality with guaranteed initialization order."""
    data = request.getfixturevalue(arrays)

    mmd = maximum_mean_discrepancy(data.X, data.Y)
    assert mmd > 0, "MMD should be positive for different distributions"

    mmd_same = maximum_mean_discrepancy(data.X, data.X_same)
    assert mmd_same < 0.1, "MMD should be near 0 for identical distributions"

    mmd_xy = maximum_mean_discrepancy(data.X, data.Y)
    mmd_yx = maximum_mean_discrepancy(data.Y, data.X)

    if torch.is_tensor(data.X):
        assert torch.allclose(mmd_xy, mmd_yx, rtol=1e-5)
    else:
        assert_allclose(mmd_xy, mmd_yx, rtol=1e-5)


@pytest.mark.parametrize("arrays", ["numpy_arrays", "torch_arrays"])
def test_mmd_validation(request, arrays):
    """Test inumpyut validation with single points."""
    data = request.getfixturevalue(arrays)

    with pytest.raises(ValueError):
        _ = maximum_mean_discrepancy(data.X[:1], data.Y[:1])

    mmd = maximum_mean_discrepancy(data.X_large, data.Y_small)
    assert numpy.isfinite(float(mmd))


@pytest.mark.parametrize("arrays", ["numpy_arrays", "torch_arrays"])
def test_mmd_broadcasting(request, arrays):
    """Test MMD with batched inputs."""
    data = request.getfixturevalue(arrays)

    # Create batched data (2, B, N, D)
    if torch.is_tensor(data.X):
        X_batch = data.X[None, None, :, :].repeat(2, 3, 1, 1)
        Y_batch = data.Y[None, None, :, :].repeat(2, 3, 1, 1)
    else:
        X_batch = numpy.ones((2, 3) + data.X.shape) * data.X
        Y_batch = numpy.ones((2, 3) + data.Y.shape) * data.Y

    mmd = maximum_mean_discrepancy(X_batch, Y_batch)
    assert mmd.shape == (2, 3), f"Expected shape (2, 3), got {mmd.shape}"

    # Check each batch independently matches unbatched computation
    for i in range(2):
        for j in range(3):
            single_mmd = maximum_mean_discrepancy(X_batch[i, j], Y_batch[i, j])

            if torch.is_tensor(data.X):
                assert torch.allclose(mmd[i, j], single_mmd)
            else:
                assert numpy.allclose(mmd[i, j], single_mmd)


def test_mmd_hamming():
    """Test MMD with Hamming distance on string arrays."""
    # Create simple arrays of equal-length strings
    rng = numpy.random.default_rng(42)
    n_samples = 3

    # Generate random DNA sequences for efficient Hamming comparison
    X = numpy.array(
        ["".join(rng.choice(["A", "T", "G", "C"], 32)) for _ in range(n_samples)]
    ).reshape(-1, 1)
    Y = numpy.array(
        ["".join(rng.choice(["A", "R", "G", "C"], 32)) for _ in range(n_samples * 2)]
    ).reshape(-1, 1)

    def hamming_distance(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        """
        Compute pairwise Hamming distances between arrays of strings.

        Args:
            x: First array of strings, shape (n, 1)
            y: Second array of strings, shape (m, 1)

        Returns:
            Distance matrix of shape (n, m)
        """
        # Reshape inputs to flatten them if needed
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)

        n, m = len(x_flat), len(y_flat)
        distances = numpy.zeros((n, m), dtype=numpy.float64)

        # Compute pairwise distances
        for i in range(n):
            for j in range(m):
                # For string arrays, count character differences
                distances[i, j] = sum(
                    1.0 for a, b in zip(x_flat[i], y_flat[j], strict=False) if a != b
                )

        return distances

    mmd = maximum_mean_discrepancy(X, Y, distance_fn=hamming_distance)
    print(mmd)
    assert mmd > 0, "MMD should be positive for different string distributions"
