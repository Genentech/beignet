from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, ReLU

from ._resnet_block import ResNetBlock


class ResNet(Module):
    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Parameters
        ----------
        c_in : int
            Input channel dimension

        c_hidden : int
            Hidden channel dimension

        no_blocks : int
            Number of resnet blocks

        no_angles : int
            Number of torsion angles to generate

        epsilon : float
            Small constant for normalization
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = ModuleList()

        for _ in range(self.no_blocks):
            layer = ResNetBlock(c_hidden=self.c_hidden)

            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)

        self.relu = ReLU()

    def forward(self, s: Tensor, s_initial: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for layer in self.layers:
            s = layer(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s
