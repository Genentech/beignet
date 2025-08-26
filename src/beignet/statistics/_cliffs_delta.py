import torch
from torch import Tensor


def cliffs_delta(
    x: Tensor,
    y: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute Cliff's delta, a non-parametric effect size measure.

    Cliff's delta is a robust non-parametric effect size that measures the
    degree of overlap between two distributions. It represents the probability
    that a randomly selected value from one distribution is greater than a
    randomly selected value from the other distribution, minus the probability
    of the reverse.

    When to Use
    -----------
    **Traditional Statistics:**
    - Robust alternative to Cohen's d when normality assumptions are violated
    - Ordinal data or ranked measurements
    - Presence of outliers that would distort parametric effect sizes
    - Small sample sizes where distributional assumptions are questionable

    **Machine Learning Contexts:**
    - Model comparison with non-normal performance metrics
    - Feature importance assessment with ordinal or skewed features
    - A/B testing with non-parametric outcomes (e.g., user rankings, ratings)
    - Fairness assessment: comparing outcomes across demographic groups
    - Anomaly detection: measuring separation between normal and anomalous data
    - Time series: comparing distributions across different time periods
    - Recommendation systems: comparing user preference distributions
    - Computer vision: comparing pixel intensity or feature distributions
    - Natural language processing: comparing text feature distributions

    **Choose Cliff's delta over Cohen's d when:**
    - Data is ordinal or heavily skewed
    - Presence of outliers
    - Non-normal distributions
    - Small sample sizes (n < 30)
    - Need assumption-free effect size measure

    **Interpretation Guidelines:**
    - |δ| < 0.147: negligible effect
    - 0.147 ≤ |δ| < 0.33: small effect
    - 0.33 ≤ |δ| < 0.474: medium effect
    - |δ| ≥ 0.474: large effect

    Parameters
    ----------
    x : Tensor, shape=(..., N)
        First sample values.
    y : Tensor, shape=(..., M)
        Second sample values.

    Returns
    -------
    output : Tensor, shape=(...)
        Cliff's delta values. Range is [-1, 1] where:
        * -1: all values in x are smaller than all values in y
        *  0: distributions completely overlap
        * +1: all values in x are larger than all values in y

    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> y = torch.tensor([4.0, 5.0, 6.0])
    >>> cliffs_delta(x, y)
    tensor(-1.0000)

    >>> x = torch.tensor([1.0, 3.0, 5.0])
    >>> y = torch.tensor([2.0, 4.0, 6.0])
    >>> cliffs_delta(x, y)
    tensor(0.0000)

    Notes
    -----
    Cliff's delta is defined as:

    δ = (N_xy - N_yx) / (N_x × N_y)

    where:
    - N_xy = number of pairs where x[i] > y[j]
    - N_yx = number of pairs where y[j] > x[i]
    - N_x, N_y = sample sizes

    Interpretation guidelines:
    - |δ| < 0.147: negligible
    - 0.147 ≤ |δ| < 0.33: small
    - 0.33 ≤ |δ| < 0.474: medium
    - |δ| ≥ 0.474: large
    """
    x = torch.atleast_1d(torch.as_tensor(x))
    y = torch.atleast_1d(torch.as_tensor(y))

    # Ensure common dtype
    if x.dtype.is_floating_point and y.dtype.is_floating_point:
        if x.dtype == torch.float64 or y.dtype == torch.float64:
            dtype = torch.float64
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32
    x = x.to(dtype)
    y = y.to(dtype)

    # Handle batch dimensions - assume last dimension is the sample dimension
    x_expanded = x.unsqueeze(-1)  # (..., N, 1)
    y_expanded = y.unsqueeze(-2)  # (..., 1, M)

    # Count comparisons
    n_xy = torch.sum(x_expanded > y_expanded, dim=(-2, -1)).to(dtype)  # x > y
    n_yx = torch.sum(x_expanded < y_expanded, dim=(-2, -1)).to(dtype)  # x < y

    # Total number of comparisons
    n_x = torch.tensor(x.shape[-1], dtype=dtype)
    n_y = torch.tensor(y.shape[-1], dtype=dtype)
    total_comparisons = n_x * n_y

    # Cliff's delta
    delta = (n_xy - n_yx) / total_comparisons

    if out is not None:
        out.copy_(delta)
        return out
    return delta
