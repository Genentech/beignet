import math

import torch
from torch import Tensor

from ._multivariable_linear_regression_power import (
    multivariable_linear_regression_power,
)


def multivariable_linear_regression_sample_size(
    r_squared: Tensor,
    n_predictors: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required sample size for multiple linear regression.

    Calculates the sample size needed to achieve desired power for testing
    the overall model significance in multiple linear regression.

    Parameters
    ----------
    r_squared : Tensor
        Expected R² (coefficient of determination) under alternative hypothesis.
        Range is [0, 1).
    n_predictors : Tensor
        Number of predictor variables (excluding intercept).
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    output : Tensor
        Required sample size (rounded up to nearest integer).

    Examples
    --------
    >>> r_squared = torch.tensor(0.15)  # Medium effect
    >>> n_predictors = torch.tensor(3)
    >>> multivariable_linear_regression_sample_size(r_squared, n_predictors)
    tensor(77.0)

    Notes
    -----
    This function uses an iterative approach to find the sample size that
    achieves the desired power for the overall F-test in multiple regression.

    Sample size considerations:
    - Larger R² values require smaller sample sizes
    - More predictors require larger sample sizes
    - Common rule of thumb: n ≥ 50 + 8p (where p = n_predictors)
    """
    r_squared = torch.atleast_1d(torch.as_tensor(r_squared))
    n_predictors = torch.atleast_1d(torch.as_tensor(n_predictors))

    # Ensure floating point dtype
    if r_squared.dtype == torch.float64 or n_predictors.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32
    r_squared = r_squared.to(dtype)
    n_predictors = n_predictors.to(dtype)

    # Validate inputs
    r_squared = torch.clamp(r_squared, min=1e-8, max=0.99)
    n_predictors = torch.clamp(n_predictors, min=1.0)

    # Initial approximation using Cohen's formula
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    # Effect size f² = R²/(1-R²)
    f_squared = r_squared / (1 - r_squared)

    # Initial approximation: n ≈ (z_α + z_β)²/f² + p + 1
    n_init = ((z_alpha + z_beta) ** 2) / f_squared + n_predictors + 1
    n_init = torch.clamp(n_init, min=n_predictors + 10)

    # Iterative refinement
    n_current = n_init
    for _ in range(15):
        # Calculate current power
        current_power = multivariable_linear_regression_power(
            r_squared, n_current, n_predictors, alpha=alpha
        )

        # Adjust sample size based on power gap
        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.1 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=n_predictors + 10, max=1e6)

    # Round up to nearest integer
    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
