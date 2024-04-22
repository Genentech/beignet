from typing import Any, Callable

import torch
from torch import Tensor


def _to_square_metric_fn(
    fn: Callable[[Tensor, Tensor, Any], Tensor],
) -> Callable[[Tensor, Tensor, Any], Tensor]:
    for dimension in range(1, 4):
        try:
            positions = torch.rand([dimension], dtype=torch.float32)

            distances = fn(positions, positions, t=0)  # type: ignore[no-untyped-def]

            if distances.ndim == 0:

                def square_metric(a: Tensor, b: Tensor, **kwargs) -> Tensor:
                    return torch.square(fn(a, b, **kwargs))
            else:

                def square_metric(a: Tensor, b: Tensor, **kwargs) -> Tensor:
                    return torch.sum(torch.square(fn(a, b, **kwargs)), dim=-1)

            return square_metric
        except TypeError:
            continue
        except ValueError:
            continue

    raise ValueError
