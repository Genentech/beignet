import torch
from torch import Tensor

import beignet.distributions


def independent_t_test_power(
    input: Tensor,
    nobs1: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: Tensor | None = None,
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
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default 'two-sided'
        Alternative hypothesis ("two-sided", "greater", "less").
    ratio : Tensor | None, optional
        Sample size ratio.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    input = torch.atleast_1d(torch.as_tensor(input))

    sample_size_group_1 = torch.atleast_1d(torch.as_tensor(nobs1))
    if ratio is None:
        ratio = torch.tensor(1.0)
    else:
        ratio = torch.atleast_1d(torch.as_tensor(ratio))

    if (
        input.dtype == torch.float64
        or sample_size_group_1.dtype == torch.float64
        or ratio.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    input = input.to(dtype)

    sample_size_group_1 = sample_size_group_1.to(dtype)

    ratio = ratio.to(dtype)

    input = torch.clamp(input, min=0.0)

    sample_size_group_1 = torch.clamp(sample_size_group_1, min=2.0)

    ratio = torch.clamp(ratio, min=0.1, max=10.0)

    sample_size_group_2 = sample_size_group_1 * ratio

    total_sample_size = sample_size_group_1 + sample_size_group_2

    degrees_of_freedom = total_sample_size - 2

    degrees_of_freedom = torch.clamp(degrees_of_freedom, min=1.0)

    se_factor = torch.sqrt(1 / sample_size_group_1 + 1 / sample_size_group_2)

    noncentrality = input / se_factor

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
        z_upper = (t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=1e-10,
        )

        z_lower = (-t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=1e-10,
        )

        power = 0.5 * (1 - torch.erf(z_upper / torch.sqrt(torch.tensor(2.0)))) + 0.5 * (
            1 + torch.erf(z_lower / torch.sqrt(torch.tensor(2.0)))
        )
    elif alt == "greater":
        z_score = (t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=1e-10,
        )

        power = 0.5 * (1 - torch.erf(z_score / torch.sqrt(torch.tensor(2.0))))
    else:
        z_score = (-t_critical - mean_nct) / torch.clamp(
            standard_deviation_nct,
            min=1e-10,
        )

        power = 0.5 * (1 + torch.erf(z_score / torch.sqrt(torch.tensor(2.0))))

    result = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(result)
        return out
    return result
