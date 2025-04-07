from typing import List

import torch
from torch.nn import Linear, Module, Parameter, Softmax, Softplus

from ._monomer_point_projection import PointProjection


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class InvariantPointAttention(Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Parameters
        ----------
        c_s: int
            Single representation channel dimension

        c_z: int
            Pair representation channel dimension

        c_hidden: int
            Hidden channel dimension

        no_heads: int
            Number of attention heads

        no_qk_points: int
            Number of query/key points to generate

        no_v_points: int
            Number of value points to generate

        inf: float
            Large number used for attention masking

        eps: float
            Small number used in angle resnet normalization
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        self.hc = self.c_hidden * self.no_heads

        self.linear_q_points = PointProjection(
            self.c_s,
            self.no_qk_points,
            self.no_heads,
            self.is_multimer,
        )

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = Parameter(torch.zeros(no_heads))

        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )

        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = Softmax(dim=-1)
        self.softplus = Softplus()
