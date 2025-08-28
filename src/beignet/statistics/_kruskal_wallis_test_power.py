import math

import torch
from torch import Tensor

import beignet.distributions


def kruskal_wallis_test_power(
    input: Tensor,
    sample_sizes: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor

    sample_sizes : Tensor

    alpha : float, default 0.05

    out : Tensor | None

    Returns
    -------
    Tensor
    """
    input = torch.atleast_1d(input)
    sample_sizes = torch.atleast_1d(sample_sizes)

    dtype = torch.promote_types(input.dtype, sample_sizes.dtype)

    input = input.to(dtype)
    sample_sizes = sample_sizes.to(dtype)

    input = torch.clamp(input, min=0.0)
    sample_sizes = torch.clamp(sample_sizes, min=2.0)

    groups = torch.tensor(sample_sizes.shape[-1], dtype=dtype)
    n = torch.sum(sample_sizes, dim=-1)
    df = groups - 1

    lambda_nc = 12 * n * input / (n + 1)

    chi2_dist = beignet.distributions.Chi2(df)
    chi2_critical = chi2_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    nc_chi2_dist = beignet.distributions.NonCentralChi2(df, lambda_nc)

    z = (chi2_critical - nc_chi2_dist.mean) / torch.clamp(
        torch.sqrt(nc_chi2_dist.variance),
        min=1e-12,
    )

    output = 0.5 * (1 - torch.erf(z / math.sqrt(2.0)))
    output = torch.clamp(output, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
