import math

import torch
from torch import Tensor


def two_way_analysis_of_variance_power(
    effect_size: Tensor,
    sample_size_per_cell: Tensor,
    levels_factor_a: Tensor,
    levels_factor_b: Tensor,
    alpha: float = 0.05,
    effect_type: str = "main_a",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power analysis for two-way ANOVA.

    Tests main effects and interaction in a two-way factorial design using
    the F-test.

    Parameters
    ----------
    effect_size : Tensor
        Cohen's f for the specific effect being tested.
    sample_size_per_cell : Tensor
        Sample size per cell in the factorial design.
    levels_factor_a : Tensor
        Number of levels for Factor A.
    levels_factor_b : Tensor
        Number of levels for Factor B.
    alpha : float, default=0.05
        Significance level.
    effect_type : {"main_a", "main_b", "interaction"}, default="main_a"
        Which effect to test:
        - "main_a": Main effect of Factor A
        - "main_b": Main effect of Factor B
        - "interaction": A × B interaction effect

    Returns
    -------
    output : Tensor, shape=(...)
        Statistical power values in range [0, 1].

    Examples
    --------
    >>> effect_size = torch.tensor(0.25)  # Medium effect
    >>> sample_size_per_cell = torch.tensor(20)
    >>> levels_factor_a = torch.tensor(2)
    >>> levels_factor_b = torch.tensor(3)
    >>> two_way_analysis_of_variance_power(effect_size, sample_size_per_cell,
    ...                                   levels_factor_a, levels_factor_b)
    tensor(0.8456)

    Notes
    -----
    Two-way ANOVA model: Y_ijk = μ + α_i + β_j + (αβ)_ij + ε_ijk

    The F-test statistics are:
    - Main A: F_A = MS_A / MS_error
    - Main B: F_B = MS_B / MS_error
    - Interaction: F_AB = MS_AB / MS_error

    Degrees of freedom:
    - Main A: df_A = a - 1
    - Main B: df_B = b - 1
    - Interaction: df_AB = (a-1)(b-1)
    - Error: df_error = ab(n-1) where n = sample_size_per_cell

    Effect size interpretation (Cohen's f):
    - f = 0.10: small effect
    - f = 0.25: medium effect
    - f = 0.40: large effect

    References
    ----------
    Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
    Lawrence Erlbaum Associates.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size_per_cell = torch.atleast_1d(torch.as_tensor(sample_size_per_cell))
    levels_factor_a = torch.atleast_1d(torch.as_tensor(levels_factor_a))
    levels_factor_b = torch.atleast_1d(torch.as_tensor(levels_factor_b))

    # Ensure floating point dtype
    dtypes = [
        effect_size.dtype,
        sample_size_per_cell.dtype,
        levels_factor_a.dtype,
        levels_factor_b.dtype,
    ]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size_per_cell = sample_size_per_cell.to(dtype)
    levels_factor_a = levels_factor_a.to(dtype)
    levels_factor_b = levels_factor_b.to(dtype)

    # Validate inputs
    effect_size = torch.clamp(effect_size, min=0.0)
    sample_size_per_cell = torch.clamp(sample_size_per_cell, min=2.0)
    levels_factor_a = torch.clamp(levels_factor_a, min=2.0)
    levels_factor_b = torch.clamp(levels_factor_b, min=2.0)

    # Total sample size
    total_n = sample_size_per_cell * levels_factor_a * levels_factor_b

    # Degrees of freedom based on effect type
    if effect_type == "main_a":
        df_num = levels_factor_a - 1
    elif effect_type == "main_b":
        df_num = levels_factor_b - 1
    elif effect_type == "interaction":
        df_num = (levels_factor_a - 1) * (levels_factor_b - 1)
    else:
        raise ValueError("effect_type must be 'main_a', 'main_b', or 'interaction'")

    df_den = levels_factor_a * levels_factor_b * (sample_size_per_cell - 1)
    df_den = torch.clamp(df_den, min=1.0)

    # Noncentrality parameter
    lambda_nc = total_n * effect_size**2

    # Critical F-value using chi-square approximation
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    chi2_critical = df_num + z_alpha * torch.sqrt(2 * df_num)
    f_critical = chi2_critical / df_num

    # Noncentral F approximation
    # Under H₁: F ~ F(df_num, df_den, λ)
    # Approximate using noncentral chi-square moments
    mean_nc_chi2 = df_num + lambda_nc
    var_nc_chi2 = 2 * (df_num + 2 * lambda_nc)
    mean_f = mean_nc_chi2 / df_num
    var_f = var_nc_chi2 / (df_num**2)

    # Finite sample adjustment for denominator df
    var_f = var_f * ((df_den + 2) / torch.clamp(df_den, min=1.0))

    std_f = torch.sqrt(torch.clamp(var_f, min=1e-12))

    # Standardized test statistic
    z_score = (f_critical - mean_f) / std_f

    # Power = P(F > F_critical | H₁)
    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    # Clamp to valid range
    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
