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

    dtype = torch.float32
    for tensor in (group_means, pooled_std):
        dtype = torch.promote_types(dtype, tensor.dtype)

    group_means = group_means.to(dtype)

    pooled_std = pooled_std.to(dtype)

    result = torch.std(group_means, dim=-1, unbiased=False) / torch.where(
        torch.abs(pooled_std) < 1e-10,
        torch.tensor(1e-10, dtype=dtype, device=pooled_std.device),
        pooled_std,
    )

    if out is not None:
        out.copy_(result)

        return out

