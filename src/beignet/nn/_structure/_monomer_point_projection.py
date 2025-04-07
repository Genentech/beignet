from typing import Tuple

import torch
from torch import Tensor

from .__point_projection import PointProjection


class MonomerPointProjection(PointProjection):
    def __init__(
        self,
        c_hidden: int,
        num_points: int,
        no_heads: int,
        return_local_points: bool = False,
    ):
        super().__init__(
            c_hidden,
            num_points,
            no_heads,
            return_local_points,
        )

    def forward(
        self,
        activations: Tensor,
        rigids,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        y = self.linear(activations)

        output_shape = y.shape[:-1] + (self.no_heads, self.num_points, 3)

        y = torch.split(y, y.shape[-1] // 3, dim=-1)

        y = torch.stack(y, dim=-1).view(output_shape)

        x = rigids[..., None, None].apply(y)

        if self.return_local_points:
            return x, y

        return x
