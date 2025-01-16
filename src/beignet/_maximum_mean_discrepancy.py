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
    else:
        xp = X.__array_namespace__()  # Get array namespace for API operations

    if X.shape[0] < 2 or Y.shape[0] < 2:
        raise ValueError("Each distribution must have at least 2 samples")

    if distance_fn is None and hasattr(xp, "expand_dims"):

        def distance_fn(x, y):
            # Broadcasting using array API operations
            diff = xp.expand_dims(x, 1) - xp.expand_dims(y, 0)
            return xp.sqrt((diff**2).sum(-1))
    elif distance_fn is None and hasattr(xp, "unsqueeze"):

        def distance_fn(x, y):
            diff = xp.unsqueeze(x, 1) - xp.unsqueeze(y, 0)
            return xp.sqrt((diff**2).sum(-1))
    else:
        raise ValueError("Array namespace does not conform to expected API")

    # Compute kernel matrices
    D_XX = distance_fn(X, X)
    D_YY = distance_fn(Y, Y)
    D_XY = distance_fn(X, Y)

    if kernel_width is None:
        # Concatenate and compute median using array API
        all_distances = xp.concat(
            [xp.reshape(D_XX, (-1,)), xp.reshape(D_YY, (-1,)), xp.reshape(D_XY, (-1,))],
            axis=0,
        )
        sq_kernel_width = xp.median(all_distances) ** 2

    # Apply RBF kernel using array API operations
    K_XX = xp.exp(-0.5 * D_XX**2 / sq_kernel_width)
    K_YY = xp.exp(-0.5 * D_YY**2 / sq_kernel_width)
    K_XY = xp.exp(-0.5 * D_XY**2 / sq_kernel_width)

    m = X.shape[-2]
    n = Y.shape[-2]

    # Compute MMD^2 with diagonal correction using array API operations
    mmd_squared = (
        (xp.sum(K_XX) - xp.trace(K_XX)) / (m * (m - 1))
        + (xp.sum(K_YY) - xp.trace(K_YY)) / (n * (n - 1))
        - 2 * xp.mean(K_XY)
    )

    if xp is torch:
        return mmd_squared.clamp_min(0.0).sqrt()

    return xp.sqrt(xp.maximum(mmd_squared, 0.0))
