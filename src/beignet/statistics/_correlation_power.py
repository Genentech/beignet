import torch
from torch import Tensor


def correlation_power(
    r: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for correlation tests.

    Statistical power is the probability of correctly rejecting a false null
    hypothesis that the population correlation is zero.

    This function is differentiable with respect to both r and sample_size
    parameters. While traditional correlation power analysis doesn't require
    gradients, differentiability enables integration into machine learning
    workflows where correlation strengths may be learned parameters.

    When to Use
    -----------
    **Traditional Statistics:**
    - Testing linear relationships between continuous variables
    - Validating theoretical models predicting correlations
    - Quality assessment in measurement instruments
    - Behavioral science studies examining variable relationships

    **Machine Learning Contexts:**
    - Feature selection: determining if features correlate significantly with targets
    - Model validation: testing if model predictions correlate with ground truth
    - Data quality assessment: detecting spurious correlations in datasets
    - Representation learning: validating that learned embeddings capture expected relationships
    - A/B testing: measuring correlations between interventions and outcomes
    - Time series analysis: detecting significant autocorrelations
    - Recommender systems: validating user-item preference correlations
    - Causal inference: testing instrument validity through correlation analysis
    - Hyperparameter optimization: correlation between parameters and performance

    **Choose correlation tests when:**
    - Both variables are continuous
    - Relationship is expected to be linear (use rank correlation for non-linear)
    - Data points are independent
    - Variables are approximately normally distributed (for parametric inference)

    **Interpretation Guidelines:**
    - |r| = 0.1: small correlation
    - |r| = 0.3: medium correlation
    - |r| = 0.5: large correlation

    Parameters
    ----------
    r : Tensor
        Expected correlation coefficient. Can be a scalar or tensor.

    sample_size : Tensor
        Sample size (total number of observations).

    alpha : float, default=0.05
        Significance level (Type I error rate).

    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Statistical power (probability between 0 and 1).

    Examples
    --------
    >>> r = torch.tensor(0.3)
    >>> n = torch.tensor(50)
    >>> correlation_power(r, n)
    tensor(0.5704)
    """
    r = torch.atleast_1d(torch.as_tensor(r))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    if r.dtype != sample_size.dtype:
        if r.dtype == torch.float64 or sample_size.dtype == torch.float64:
            r = r.to(torch.float64)
            sample_size = sample_size.to(torch.float64)
        else:
            r = r.to(torch.float32)
            sample_size = sample_size.to(torch.float32)

    epsilon = 1e-7
    r_clamped = torch.clamp(r, -1 + epsilon, 1 - epsilon)
    z_r = 0.5 * torch.log((1 + r_clamped) / (1 - r_clamped))

    se_z = 1.0 / torch.sqrt(sample_size - 3)

    z_stat = z_r / se_z

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError(f"Unknown alternative: {alternative}")

    if alt == "two-sided":
        z_alpha_2 = torch.erfinv(
            torch.tensor(1 - alpha / 2, dtype=r.dtype)
        ) * torch.sqrt(torch.tensor(2.0, dtype=r.dtype))

        cdf_upper = 0.5 * (
            1
            + torch.erf(
                (z_alpha_2 - z_stat) / torch.sqrt(torch.tensor(2.0, dtype=r.dtype))
            )
        )
        cdf_lower = 0.5 * (
            1
            + torch.erf(
                (-z_alpha_2 - z_stat) / torch.sqrt(torch.tensor(2.0, dtype=r.dtype))
            )
        )
        power = 1 - (cdf_upper - cdf_lower)

    elif alt == "greater":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=r.dtype)) * torch.sqrt(
            torch.tensor(2.0, dtype=r.dtype)
        )

        power = 1 - 0.5 * (
            1
            + torch.erf(
                (z_alpha - z_stat) / torch.sqrt(torch.tensor(2.0, dtype=r.dtype))
            )
        )

    elif alt == "less":
        z_alpha = torch.erfinv(torch.tensor(alpha, dtype=r.dtype)) * torch.sqrt(
            torch.tensor(2.0, dtype=r.dtype)
        )

        power = 0.5 * (
            1
            + torch.erf(
                (z_alpha - z_stat) / torch.sqrt(torch.tensor(2.0, dtype=r.dtype))
            )
        )
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
