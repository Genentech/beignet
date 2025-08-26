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
    n_y = torch.tensor(y.shape[-1], dtype=dtype)
    if n_y < 2:
        raise ValueError("Control group (y) must have at least 2 observations")

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
