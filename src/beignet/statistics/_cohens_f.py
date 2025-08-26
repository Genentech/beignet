import torch
from torch import Tensor


def cohens_f(
    group_means: Tensor,
    pooled_std: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    group_means = torch.atleast_1d(torch.as_tensor(group_means))

    pooled_std = torch.atleast_1d(torch.as_tensor(pooled_std))

    if group_means.dtype == torch.float64 or pooled_std.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    group_means = group_means.to(dtype)

    pooled_std = pooled_std.to(dtype)

    output = torch.std(group_means, dim=-1, unbiased=False) / torch.where(
        torch.abs(pooled_std) < 1e-10,
        torch.tensor(1e-10, dtype=dtype, device=pooled_std.device),
        pooled_std,
    )

    if out is not None:
        out.copy_(output)

        return out

    return output
