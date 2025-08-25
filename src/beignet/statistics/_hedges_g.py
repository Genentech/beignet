from torch import Tensor

from beignet.statistics._cohens_d import cohens_d


def hedges_g(
    group1: Tensor,
    group2: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute Hedges' g effect size between two groups.

    Hedges' g is a bias-corrected version of Cohen's d that provides a more
    accurate estimate of effect size, especially for small sample sizes.

    This function is fully differentiable with respect to both group inputs.
    While differentiability is not essential for typical statistical use cases,
    it allows integration into machine learning workflows where effect sizes
    are computed from learned features or used in differentiable experimental
    design optimization.

    Parameters
    ----------
    group1 : Tensor, shape (..., N1)
        First group of observations. The last dimension contains the observations.

    group2 : Tensor, shape (..., N2)
        Second group of observations. The last dimension contains the observations.

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor, shape (...)
        Hedges' g effect size. Positive values indicate group1 > group2.

    Examples
    --------
    >>> group1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> group2 = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
    >>> hedges_g(group1, group2)
    tensor(-0.6804)
    """
    # First compute Cohen's d
    cohens_d_value = cohens_d(group1, group2, pooled=True)

    # Sample sizes
    n1 = group1.shape[-1]
    n2 = group2.shape[-1]

    # Degrees of freedom
    df = n1 + n2 - 2

    # Bias correction factor (Hedges & Olkin, 1985)
    # J(df) = Gamma((df+1)/2) / (sqrt(df/2) * Gamma(df/2))
    # Approximation: J(df) ≈ 1 - 3/(4*df - 1)
    correction_factor = 1.0 - 3.0 / (4.0 * df - 1.0)

    # Apply correction
    output = cohens_d_value * correction_factor

    if out is not None:
        out.copy_(output)
        return out

    return output
