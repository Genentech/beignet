import math

import torch
from torch import Tensor


# ref https://github.com/rusty1s/pytorch_cluster/blob/master/torch_cluster/radius.py
def radius(
    x: Tensor,
    y: Tensor,
    r: float,
    batch_x: Tensor | None = None,
    batch_y: Tensor | None = None,
    ignore_same_index: bool = False,
    chunk_size: int | None = None,
) -> Tensor:
    """For each element in `y` find all points in `x` within distance `r`"""
    N, _ = x.shape
    M, _ = y.shape

    if chunk_size is None:
        chunk_size = M + 1

    if batch_x is None:
        batch_x = torch.zeros(N, dtype=torch.int64, device=x.device)

    if batch_y is None:
        batch_y = torch.zeros(M, dtype=torch.int64, device=x.device)

    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    batch_size = int(batch_x.max()) + 1
    batch_size = max(batch_size, int(batch_y.max()) + 1)
    assert batch_size > 0

    r2 = torch.as_tensor(r * r, dtype=x.dtype, device=x.device)

    n_chunks = math.ceil(M / chunk_size)

    rows = []
    cols = []

    for _, (y_chunk, batch_y_chunk, index_y_chunk) in enumerate(
        zip(
            torch.chunk(y, n_chunks),
            torch.chunk(batch_y, n_chunks),
            torch.chunk(torch.arange(M, device=x.device), n_chunks),
            strict=False,
        )
    ):
        # [M_chunk, N]
        pdist = (y_chunk[:, None] - x[None]).pow(2).sum(dim=-1)

        same_batch = batch_y_chunk[:, None] == batch_x[None]
        same_index = index_y_chunk[:, None] == torch.arange(N, device=x.device)[None]

        connected = (pdist <= r2) & same_batch
        if ignore_same_index:
            connected = connected & ~same_index

        row, col = torch.nonzero(connected, as_tuple=True)
        rows.append(row + index_y_chunk[0])
        cols.append(col)

    row = torch.cat(rows, dim=0)
    col = torch.cat(cols, dim=0)

    return torch.stack((row, col), dim=0)


def radius_graph(
    x: Tensor,
    r: float,
    batch: Tensor | None = None,
    chunk_size: int | None = None,
    loop: bool = False,
) -> Tensor:
    return radius(
        x, x, r, batch, batch, ignore_same_index=not loop, chunk_size=chunk_size
    )
