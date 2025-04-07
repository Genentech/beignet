import math
from typing import List, Sequence

import torch
from torch import Tensor
from torch.nn import Linear, Parameter, Softmax, Softplus

from .__invariant_point_attention import InvariantPointAttention
from ._multimer_point_projection import MultimerPointProjection


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


class MultimerInvariantPointAttention(InvariantPointAttention):
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
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super().__init__(
            c_s,
            c_z,
            c_hidden,
            no_heads,
            no_qk_points,
            no_v_points,
            inf,
            eps,
        )

        self.linear_k = Linear(self.c_s, self.hc, bias=False)
        self.linear_v = Linear(self.c_s, self.hc, bias=False)

        self.linear_k_points = MultimerPointProjection(
            self.c_s,
            self.no_qk_points,
            self.no_heads,
        )

        self.linear_v_points = MultimerPointProjection(
            self.c_s,
            self.no_v_points,
            self.no_heads,
        )

        # foo

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = Parameter(torch.zeros((no_heads)))

        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )

        self.linear_out = Linear(
            concat_out_dim,
            self.c_s,
            init="final",
        )

        self.softmax = Softmax(dim=-1)
        self.softplus = Softplus()

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        r,  #: Rigid | Rigid3Array,
        mask: Tensor,
        _z_references: Sequence[Tensor] | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        s: Tensor
            [*, N_res, C_s] single representation

        z: Tensor
            [*, N_res, N_res, C_z] pair representation

        r: Rigid | Rigid3Array
            [*, N_res] transformation object

        mask: Tensor
            [*, N_res] mask

        Returns
        -------
        Tensor
            [*, N_res, C_s] single representation update
        """
        z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, P_qk]
        q_pts = self.linear_q_points(s, r)

        # The following two blocks are equivalent
        # They're separated only to preserve compatibility with old AF weights

        # [*, N_res, H * C_hidden]
        k = self.linear_k(s)
        v = self.linear_v(s)

        # [*, N_res, H, C_hidden]
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, P_qk, 3]
        k_pts = self.linear_k_points(s, r)

        # [*, N_res, H, P_v, 3]
        v_pts = self.linear_v_points(s, r)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, [1, 0, 2]),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, [1, 2, 0]),  # [*, H, C_hidden, N_res]
        )

        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, [2, 0, 1])

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)

        pt_att = pt_att**2

        pt_att = sum(torch.unbind(pt_att, dim=-1))

        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )

        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, [2, 0, 1])

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        o_pt = (
            a[..., None, :, :, None]
            * permute_final_dims(v_pts, [1, 3, 0, 2])[..., None, :, :]
        )
        o_pt = torch.sum(o_pt, dim=-2)

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, [2, 0, 3, 1])
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps)
        o_pt_norm = flatten_final_dims(o_pt_norm, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
        o_pt = torch.unbind(o_pt, dim=-1)

        # [*, N_res, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = torch.concatenate((o, *o_pt, o_pt_norm, o_pair), dim=-1)
        s = s.to(dtype=z[0].dtype)
        s = self.linear_out(s)

        return s
