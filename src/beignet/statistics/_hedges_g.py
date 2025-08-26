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

    When to Use
    -----------
    **Traditional Statistics:**
    - Small sample sizes (n < 20 per group) where bias correction matters
    - Meta-analyses requiring most accurate effect size estimates
    - Clinical trials with limited participants
    - Pilot studies with small samples

    **Machine Learning Contexts:**
    - A/B testing with small user samples
    - Few-shot learning: comparing performance with limited data
    - Medical AI: small patient cohorts in rare disease studies
    - Early-stage model development with limited training data
    - Cross-validation with small validation sets
    - Transfer learning: small target domain samples
    - Active learning: effect sizes with incrementally growing datasets
    - Federated learning: local client effects with small data

    **Choose Hedges' g over Cohen's d when:**
    - Small sample sizes (especially n < 20 per group)
    - Need most unbiased effect size estimate
    - Meta-analysis combining studies with varying sample sizes
    - Reporting in publications emphasizing methodological rigor

    **Interpretation Guidelines:**
    - Same as Cohen's d: |g| = 0.2 (small), 0.5 (medium), 0.8 (large)
    - More accurate than Cohen's d for small samples

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
    sample_size_group_1 = group1.shape[-1]
    sample_size_group_2 = group2.shape[-1]

    # Degrees of freedom
    degrees_of_freedom = sample_size_group_1 + sample_size_group_2 - 2

    # Bias correction factor (Hedges & Olkin, 1985)
    # J(degrees_of_freedom) = Gamma((degrees_of_freedom+1)/2) / (sqrt(degrees_of_freedom/2) * Gamma(degrees_of_freedom/2))
    # Approximation: J(degrees_of_freedom) ≈ 1 - 3/(4*degrees_of_freedom - 1)
    correction_factor = 1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)

    # Apply correction
    output = cohens_d_value * correction_factor

    if out is not None:
        out.copy_(output)
        return out

    return output
