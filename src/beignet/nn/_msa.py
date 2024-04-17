import torch
from torch import Tensor
from torch.nn import Conv1d, Module

import beignet.operators


class MSA(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 512,
        kernel_size: int = 18,
        *,
        gap_penalty: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.gap_penalty = gap_penalty

        self.temperature = temperature

        self.embedding = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
        )

    def forward(self, inputs: (Tensor, Tensor)) -> Tensor:
        matrices, shapes = inputs

        embedding = self.embedding(matrices)

        embedding = embedding @ embedding[0].T

        output = beignet.operators.needleman_wunsch(
            embedding,
            shapes,
            gap_penalty=self.gap_penalty,
            temperature=self.temperature,
        )

        return torch.einsum(
            "ja, nij -> nia",
            torch.mean(
                torch.einsum(
                    "nia, nij -> nja",
                    matrices,
                    output,
                ),
                dim=0,
            ),
            output,
        )
