import math

import torch
import torch.nn as nn
from torch import Tensor

from ._adaptive_layer_norm import AdaptiveLayerNorm


class AttentionPairBias(nn.Module):
    r"""
    Attention with pair bias and mask from AlphaFold 3 Algorithm 24.

    This implements the AttentionPairBias operation with conditioning signal support
    for diffusion models. It uses AdaLN when conditioning is provided, or standard
    LayerNorm when not. Includes proper gating and output projection.

    Parameters
    ----------
    c_a : int
        Channel dimension for input representation 'a'
    c_s : int
        Channel dimension for conditioning signal 's' (can be None if no conditioning)
    c_z : int
        Channel dimension for pair representation 'z'
    n_head : int
        Number of attention heads

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import AttentionPairBias
    >>> batch_size, seq_len = 2, 10
    >>> c_a, c_s, c_z, n_head = 256, 384, 128, 16
    >>> module = AttentionPairBias(c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head)
    >>> a = torch.randn(batch_size, seq_len, c_a)
    >>> s = torch.randn(batch_size, seq_len, c_s)
    >>> z = torch.randn(batch_size, seq_len, seq_len, c_z)
    >>> beta = torch.randn(batch_size, seq_len, seq_len, n_head)
    >>> a_out = module(a, s, z, beta)
    >>> a_out.shape
    torch.Size([2, 10, 256])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 24: AttentionPairBias with pair bias and mask
    """

    def __init__(self, c_a: int, c_s: int, c_z: int, n_head: int):
        super().__init__()

        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.n_head = n_head
        self.head_dim = c_a // n_head

        if c_a % n_head != 0:
            raise ValueError(
                f"Channel dimension {c_a} must be divisible by number of heads {n_head}"
            )

        # Input projections - Algorithm 24 steps 1-4
        # Step 1-2: If {si} ≠ ∅ then ai ← AdaLN(ai, si) else ai ← LayerNorm(ai)
        if c_s is not None:
            self.ada_ln = AdaptiveLayerNorm(c=c_a, c_s=c_s)
        else:
            self.layer_norm = nn.LayerNorm(c_a)

        # Step 6: q_i^h = Linear(ai)
        self.linear_q = nn.Linear(c_a, c_a, bias=True)

        # Step 7: k_i^h, v_i^h = LinearNoBias(ai)
        self.linear_k = nn.Linear(c_a, c_a, bias=False)
        self.linear_v = nn.Linear(c_a, c_a, bias=False)

        # Step 8: b_ij^h ← LinearNoBias(LayerNorm(zij)) + βij
        self.linear_b = nn.Linear(c_z, n_head, bias=False)
        self.layer_norm_z = nn.LayerNorm(c_z)

        # Step 9: g_i^h ← sigmoid(LinearNoBias(ai))
        self.linear_g = nn.Linear(c_a, c_a, bias=False)

        # Step 11: Output projection
        self.output_linear = nn.Linear(c_a, c_a, bias=False)

        # Scale factor for attention (Step 10: 1/√c where c = ca/Nhead)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Output projection with adaLN-Zero pattern (Steps 12-13)
        if c_s is not None:
            # Step 13: sigmoid(Linear(si, biasinit=-2.0)) ⊙ ai
            self.linear_s_gate = nn.Linear(c_s, c_a, bias=True)
            # Initialize bias to -2.0 as specified
            with torch.no_grad():
                self.linear_s_gate.bias.fill_(-2.0)

    def forward(
        self, a: Tensor, s: Tensor = None, z: Tensor = None, beta: Tensor = None
    ) -> Tensor:
        r"""
        Forward pass of attention with pair bias.

        Parameters
        ----------
        a : Tensor, shape=(..., seq_len, c_a)
            Input representation
        s : Tensor, shape=(..., seq_len, c_s), optional
            Conditioning signal (if None, uses LayerNorm instead of AdaLN)
        z : Tensor, shape=(..., seq_len, seq_len, c_z), optional
            Pair representation for computing attention bias
        beta : Tensor, shape=(..., seq_len, seq_len, n_head), optional
            Additional bias terms

        Returns
        -------
        a_out : Tensor, shape=(..., seq_len, c_a)
            Updated representation after attention with pair bias
        """
        batch_shape = a.shape[:-2]
        seq_len = a.shape[-2]

        # Algorithm 24 Steps 1-4: Input projections
        # Step 1: if {si} ≠ ∅ then
        if s is not None and hasattr(self, "ada_ln"):
            # Step 2: ai ← AdaLN(ai, si)
            a_normed = self.ada_ln(a, s)
        else:
            # Step 4: ai ← LayerNorm(ai)
            a_normed = self.layer_norm(a)

        # Step 6: q_i^h = Linear(ai)
        q = self.linear_q(a_normed)
        q = q.view(*batch_shape, seq_len, self.n_head, self.head_dim)

        # Step 7: k_i^h, v_i^h = LinearNoBias(ai)
        k = self.linear_k(a_normed)
        k = k.view(*batch_shape, seq_len, self.n_head, self.head_dim)

        v = self.linear_v(a_normed)
        v = v.view(*batch_shape, seq_len, self.n_head, self.head_dim)

        # Step 8: b_ij^h ← LinearNoBias(LayerNorm(zij)) + βij
        if z is not None:
            z_normed = self.layer_norm_z(z)
            b_z = self.linear_b(z_normed)  # Shape: (..., seq_len, seq_len, n_head)
        else:
            b_z = torch.zeros(
                *batch_shape,
                seq_len,
                seq_len,
                self.n_head,
                device=a.device,
                dtype=a.dtype,
            )

        if beta is not None:
            b = b_z + beta
        else:
            b = b_z

        # Step 9: g_i^h ← sigmoid(LinearNoBias(ai))
        g = torch.sigmoid(self.linear_g(a_normed))
        g = g.view(*batch_shape, seq_len, self.n_head, self.head_dim)

        # Step 10: A_ij^h ← softmax_j(1/√c * q_i^h * k_j^h + b_ij^h)
        # Compute attention scores: q @ k.T / √d + bias
        # q: (..., seq_len, n_head, head_dim)
        # k: (..., seq_len, n_head, head_dim)
        # Want: (..., n_head, seq_len, seq_len)
        attn_logits = (
            torch.einsum("...ihd,...jhd->...hij", q, k) * self.scale
        )  # Shape: (..., n_head, seq_len, seq_len)

        # Add bias: b has shape (..., seq_len, seq_len, n_head)
        # We need to transpose to match attention shape
        b_transposed = b.transpose(-3, -1)  # (..., n_head, seq_len, seq_len)
        attn_logits = attn_logits + b_transposed

        # Apply softmax over the last dimension (keys)
        attn_weights = torch.softmax(attn_logits, dim=-1)

        # Step 11: ai ← LinearNoBias(concat_h(g_i^h ⊙ Σ_j A_ij^h v_j^h))
        # Apply attention to values and gate
        # attn_weights: (..., n_head, seq_len, seq_len)
        # v: (..., seq_len, n_head, head_dim)
        attended_v = torch.einsum(
            "...hij,...jhd->...hid", attn_weights, v
        )  # Shape: (..., n_head, seq_len, head_dim)

        # Reshape g to match attended_v shape: (..., seq_len, n_head, head_dim) -> (..., n_head, seq_len, head_dim)
        g_reshaped = g.transpose(-3, -2)  # Shape: (..., n_head, seq_len, head_dim)

        # Apply gating: g ⊙ attended_v
        gated_output = g_reshaped * attended_v  # Element-wise multiplication

        # Concatenate heads: reshape to (..., seq_len, c_a)
        concat_output = (
            gated_output.transpose(-3, -2)
            .contiguous()
            .view(*batch_shape, seq_len, self.c_a)
        )

        # Linear projection
        a_out = self.output_linear(concat_output)

        # Algorithm 24 Steps 12-14: Output projection (from adaLN-Zero)
        # Step 12: if {si} ≠ ∅ then
        if s is not None and hasattr(self, "linear_s_gate"):
            # Step 13: ai ← sigmoid(Linear(si, biasinit=-2.0)) ⊙ ai
            s_gate = torch.sigmoid(self.linear_s_gate(s))
            a_out = s_gate * a_out

        # Step 15: return {ai}
        return a_out
