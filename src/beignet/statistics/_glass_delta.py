import torch
from torch import Tensor


def glass_delta(
    x: Tensor,
    y: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute Glass's delta, a standardized effect size measure.

    Glass's delta is similar to Cohen's d but uses only the standard deviation
    of the control group (typically y) as the standardizer, making it more
    appropriate when one group is a control condition and variances are unequal.

    When to Use
    -----------
    **Traditional Statistics:**
    - Treatment vs. control comparisons with unequal variances
    - Clinical trials where control group variance is known/stable
    - Educational interventions compared to standard curriculum
    - Pre-post designs where baseline variability is the reference

    **Machine Learning Contexts:**
    - A/B testing: comparing new model vs. baseline with stable baseline performance
    - Model improvement: measuring gains relative to production model variance
    - Transfer learning: comparing adapted model to source domain performance
    - Hyperparameter tuning: measuring improvement relative to default configuration
    - Ablation studies: comparing modified model to control architecture
    - Benchmark comparisons: measuring improvement relative to established baseline
    - Data augmentation: comparing enhanced vs. original training data
    - Feature engineering: comparing new features vs. baseline feature set

    **Choose Glass's delta over Cohen's d when:**
    - Unequal group variances (heteroscedasticity)
    - Clear control/treatment distinction
    - Control group variance is more stable/meaningful
    - Want to standardize by control group variability specifically

    **Interpretation Guidelines:**
    - Same general scale as Cohen's d: |Δ| = 0.2 (small), 0.5 (medium), 0.8 (large)
    - Interpretation relative to control group variability

    Parameters
    ----------
    x : Tensor, shape=(..., N)
        Treatment/experimental group values.
    y : Tensor, shape=(..., M)
        Control group values (used for standardization).

    Returns
    -------
    output : Tensor, shape=(...)
        Glass's delta values. Positive values indicate x > y on average.

    Examples
    --------
    >>> x = torch.tensor([3.0, 4.0, 5.0, 6.0])
    >>> y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> glass_delta(x, y)
    tensor(1.5492)

    Notes
    -----
    Glass's delta is defined as:

    Δ = (μ_x - μ_y) / σ_y

    where:
    - μ_x, μ_y are the means of groups x and y
    - σ_y is the standard deviation of the control group y

    Interpretation (similar to Cohen's d):
    - |Δ| = 0.2: small effect
    - |Δ| = 0.5: medium effect
    - |Δ| = 0.8: large effect

    Glass's delta is preferred over Cohen's d when:
    - One group is clearly a control condition
    - Group variances are substantially different
    - The control group variance is more reliable/stable
    """
    x = torch.atleast_1d(torch.as_tensor(x))
    y = torch.atleast_1d(torch.as_tensor(y))

    # Ensure common floating point dtype
    if x.dtype.is_floating_point and y.dtype.is_floating_point:
        if x.dtype == torch.float64 or y.dtype == torch.float64:
            dtype = torch.float64
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32
    x = x.to(dtype)
    y = y.to(dtype)

    # Compute means
    mean_x = torch.mean(x, dim=-1)
    mean_y = torch.mean(y, dim=-1)

    # Compute standard deviation of control group (y) with sample correction
    # Note: y.shape[-1] must be >= 2 for meaningful variance calculation
    var_y = torch.var(
        y, dim=-1, correction=1
    )  # Sample variance with Bessel's correction
    std_y = torch.sqrt(torch.clamp(var_y, min=torch.finfo(dtype).eps))

    # Glass's delta
    delta = (mean_x - mean_y) / std_y

    if out is not None:
        out.copy_(delta)
        return out
    return delta
