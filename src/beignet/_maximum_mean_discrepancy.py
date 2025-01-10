import numpy


def maximum_mean_discrepancy(
    X,
    Y,
    distance_fn=None,
    kernel_width: float | None = None,
    eps: float = 1e-16,
    rng: numpy.random.Generator | None = None,
) -> float:
    r"""
    Compute Maximum Mean Discrepancy between samples X and Y
    using a squared exponential kernel $k(x, y) = \exp(-0.5 * d(x, y)^2 / \gamma^2)$,
    where $d$ is a distance function and $\gamma$ is the kernel width.
    By default we use Euclidean distance and the median heuristic for $\gamma$.

    Args:
        X: (m, d) NumPy array of samples from first distribution
        Y: (n, d) NumPy array of samples from second distribution
        gamma: kernel width ($\sigma$ in an RBF kernel)

    Returns:
        Empirical MMD estimate
    """

    # randomly split samples in half
    n = X.shape[0]
    m = Y.shape[0]

    if n < 2 or m < 2:
        raise ValueError(
            "This function expects at least 2 samples from each distribution"
        )

    if rng is None:
        rng = numpy.random.default_rng()

    # use same rng state to ensure symmetry
    state = rng.bit_generator.state
    x_perm_idx = rng.permutation(n)
    rng.bit_generator.state = state
    y_perm_idx = rng.permutation(m)

    X1 = X[x_perm_idx[: n // 2]]
    X2 = X[x_perm_idx[n // 2 :]]

    Y1 = Y[y_perm_idx[: m // 2]]
    Y2 = Y[y_perm_idx[m // 2 :]]

    if distance_fn is None:

        def distance_fn(x, y):
            return numpy.linalg.norm(x[:, None] - y[None, :], axis=-1)

    # Compute distance matrices
    d_X1_X2 = distance_fn(X1, X2)
    d_X1_Y1 = distance_fn(X1, Y1)
    d_Y1_Y2 = distance_fn(Y1, Y2)
    d_X2_Y2 = distance_fn(X2, Y2)

    if kernel_width is None:
        # Use median heuristic for kernel width
        kernel_width = numpy.median(
            numpy.concatenate(
                [
                    d_X1_X2.flatten(),
                    d_X1_Y1.flatten(),
                    d_Y1_Y2.flatten(),
                    d_X2_Y2.flatten(),
                ]
            )
        )

    # Compute kernel matrices
    K_X1_X2 = numpy.exp(-0.5 * d_X1_X2**2 / kernel_width**2)
    K_X1_Y1 = numpy.exp(-0.5 * d_X1_Y1**2 / kernel_width**2)
    K_Y1_Y2 = numpy.exp(-0.5 * d_Y1_Y2**2 / kernel_width**2)
    K_X2_Y2 = numpy.exp(-0.5 * d_X2_Y2**2 / kernel_width**2)

    # Calculate MMD^2
    mmd_squared = (
        numpy.mean(K_X1_X2)
        + numpy.mean(K_Y1_Y2)
        - numpy.mean(K_X1_Y1)
        - numpy.mean(K_X2_Y2)
    )

    return numpy.sqrt(max(mmd_squared, eps))  # Return MMD (not squared)
