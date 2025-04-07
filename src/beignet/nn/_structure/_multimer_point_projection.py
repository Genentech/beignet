from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import Linear

from .__point_projection import PointProjection


class MonomerPointProjection(PointProjection):
    def __init__(
        self,
        c_hidden: int,
        num_points: int,
        no_heads: int,
        is_multimer: bool,
        return_local_points: bool = False,
    ):
        super().__init__()

        self.return_local_points = return_local_points
        self.no_heads = no_heads
        self.num_points = num_points
        self.is_multimer = is_multimer

        # Multimer requires this to be run with fp32 precision during training
        if self.is_multimer:
            precision = torch.float32
        else:
            precision = None

        self.linear = Linear(c_hidden, no_heads * 3 * num_points, precision=precision)

    def forward(
        self,
        activations: Tensor,
        rigids: Union[Rigid, Rigid3Array],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # TODO: Needs to run in high precision during training
        points_local = self.linear(activations)
        out_shape = points_local.shape[:-1] + (self.no_heads, self.num_points, 3)

        if self.is_multimer:
            points_local = points_local.view(
                points_local.shape[:-1] + (self.no_heads, -1)
            )

        points_local = torch.split(points_local, points_local.shape[-1] // 3, dim=-1)

        points_local = torch.stack(points_local, dim=-1).view(out_shape)

        points_global = rigids[..., None, None].apply(points_local)

        if self.return_local_points:
            return points_global, points_local

        return points_global
