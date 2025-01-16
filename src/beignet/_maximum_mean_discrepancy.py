from typing import Any, Callable, Optional

import numpy.typing as npt
import torch


def maximum_mean_discrepancy(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    distance_fn: Optional[Callable[[Any, Any], Any]] = None,
    kernel_width: Optional[float] = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy between samples X and Y
    using a squared exponential kernel k(x,y) = exp(-0.5 * d(x,y)^2 / gamma^2).
    Uses Array API standard operations for library compatibility.

    Args:
        X: (m, d) array of samples from first distribution
        Y: (n, d) array of samples from second distribution
        distance_fn: Optional callable for custom distance metric
        kernel_width: Optional float for kernel bandwidth

    Returns:
        float: Empirical MMD estimate
    """

    if torch.is_tensor(X):
        xp = torch
        xp_is_torch = True
    else:
        xp = X.__array_namespace__()  # Get array namespace for API operations
        xp_is_torch = False

    if X.shape[-2] < 2 or Y.shape[-2] < 2:
        raise ValueError("Each distribution must have at least 2 samples")

    if distance_fn is None and hasattr(xp, "expand_dims"):

        def distance_fn(x, y):
            # Broadcasting using array API operations
            diff = xp.expand_dims(x, -2) - xp.expand_dims(y, -3)
            return xp.sqrt((diff**2).sum(-1))

    elif distance_fn is None and xp_is_torch:

        def distance_fn(x, y):
            diff = xp.unsqueeze(x, -2) - xp.unsqueeze(y, -3)
            return xp.sqrt((diff**2).sum(-1))
    else:
        raise ValueError("Array namespace does not conform to expected API")

    # Compute kernel matrices
    D_XX = distance_fn(X, X)
    D_YY = distance_fn(Y, Y)
    D_XY = distance_fn(X, Y)

    batch_shape = D_XX.shape[:-2]
    if kernel_width is None:
        # Preserve all batch dimensions, flatten only distance matrices
        all_distances = xp.concat(
            [
                xp.reshape(D_XX, (*batch_shape, -1)),
                xp.reshape(D_YY, (*batch_shape, -1)),
                xp.reshape(D_XY, (*batch_shape, -1)),
            ],
            axis=-1,
        )

        if xp_is_torch:
            kernel_width = xp.median(all_distances, dim=-1).values
        else:
            kernel_width = xp.median(all_distances, axis=-1)

        # Add necessary dimensions for broadcasting
        kernel_width = xp.reshape(kernel_width, (*batch_shape, 1, 1))

    # Apply RBF kernel using array API operations
    sq_kernel_width = kernel_width**2
    K_XX = xp.exp(-0.5 * D_XX**2 / sq_kernel_width)
    K_YY = xp.exp(-0.5 * D_YY**2 / sq_kernel_width)
    K_XY = xp.exp(-0.5 * D_XY**2 / sq_kernel_width)

    m = X.shape[-2]
    n = Y.shape[-2]

    # Compute MMD^2 with diagonal correction using array API operations
    if xp_is_torch:

        def batched_trace(x, dim1=-2, dim2=-1):
            """Compute trace along last two dims while preserving batch dims."""
            i = torch.arange(x.size(dim1))
            return (
                torch.gather(x, dim2, i.expand(x.shape[:-1]).unsqueeze(-1))
                .squeeze(-1)
                .sum(-1)
            )

        # Then in the MMD computation:
        mmd_squared = (
            (K_XX.sum((-1, -2)) - batched_trace(K_XX)) / (m * (m - 1))
            + (K_YY.sum((-1, -2)) - batched_trace(K_YY)) / (n * (n - 1))
            - 2 * K_XY.mean((-1, -2))
        )

    else:
        mmd_squared = (
            (xp.sum(K_XX, axis=(-1, -2)) - xp.trace(K_XX, axis1=-1, axis2=-2))
            / (m * (m - 1))
            + (xp.sum(K_YY, axis=(-1, -2)) - xp.trace(K_YY, axis1=-1, axis2=-2))
            / (n * (n - 1))
            - 2 * xp.mean(K_XY, axis=(-1, -2))
        )

    if xp is torch:
        return mmd_squared.clamp_min(0.0).sqrt()

    return xp.sqrt(xp.maximum(mmd_squared, 0.0))
