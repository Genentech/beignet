import torch
from torch import Tensor

import beignet.distributions


def welch_t_test_power(
    input: Tensor,
    nobs1: Tensor,
    nobs2: Tensor,
    var_ratio: Tensor | float = 1.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    nobs1 : Tensor
        Sample size.
    nobs2 : Tensor
        Sample size.
    var_ratio : Tensor | float, default 1.0
        Sample size ratio.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default 'two-sided'
        Alternative hypothesis ("two-sided", "greater", "less").
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    input = torch.atleast_1d(torch.as_tensor(input))

    sample_size_group_1 = torch.atleast_1d(torch.as_tensor(nobs1))
    sample_size_group_2 = torch.atleast_1d(torch.as_tensor(nobs2))

    vr = torch.as_tensor(var_ratio)

    if any(
        t.dtype == torch.float64
        for t in (
            input,
            sample_size_group_1,
            sample_size_group_2,
            vr if isinstance(vr, Tensor) else torch.tensor(0.0),
        )
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    input = input.to(dtype)

    sample_size_group_1 = sample_size_group_1.to(dtype)
    sample_size_group_2 = sample_size_group_2.to(dtype)
    if isinstance(vr, Tensor):
        vr = vr.to(dtype)
    else:
        vr = torch.tensor(float(vr), dtype=dtype)

    input = torch.clamp(input, min=0.0)

    sample_size_group_1 = torch.clamp(sample_size_group_1, min=2.0)
    sample_size_group_2 = torch.clamp(sample_size_group_2, min=2.0)

    vr = torch.clamp(vr, min=1e-6, max=1e6)

    a = 1.0 / sample_size_group_1

    b = vr / sample_size_group_2

    se2 = a + b

    standard_error = torch.sqrt(se2)

    degrees_of_freedom = (se2**2) / (
        a**2 / torch.clamp(sample_size_group_1 - 1, min=1.0)
        + b**2 / torch.clamp(sample_size_group_2 - 1, min=1.0)
    )

    noncentrality = input / torch.clamp(standard_error, min=1e-12)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Get critical values from central t-distribution
    t_dist = beignet.distributions.StudentT(degrees_of_freedom)
    if alt == "two-sided":
        t_critical = t_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
    else:
        t_critical = t_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    # Use non-central t-distribution for power calculation
    nc_t_dist = beignet.distributions.NonCentralT(degrees_of_freedom, noncentrality)

    # Get mean and variance from the distribution
    mean_nct = nc_t_dist.mean
    variance_nct = nc_t_dist.variance
    standard_deviation_nct = torch.sqrt(variance_nct)

    if alt == "two-sided":
        zu = (t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=1e-10,
        )

        zl = (-t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=1e-10,
        )

        power = 0.5 * (
            1 - torch.erf(zu / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        ) + 0.5 * (1 + torch.erf(zl / torch.sqrt(torch.tensor(2.0, dtype=dtype))))
    elif alt == "greater":
        zscore = (t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=1e-10,
        )

        power = 0.5 * (
            1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )
    else:
        zscore = (-t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=1e-10,
        )

        power = 0.5 * (
            1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    result = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(result)

        return out

    return result
