import math

import torch
import torch.nn as nn
from torch import Tensor


class TriangleAttentionStartingNode(nn.Module):
    r"""
    Triangular gated self-attention around starting node from AlphaFold 3.

    This implements Algorithm 14 from AlphaFold 3, which performs triangular
    gated self-attention where the attention is computed around the starting
    node of each edge in the triangle.

    Parameters
    ----------
    c : int, default=32
        Channel dimension for the pair representation
    n_head : int, default=4
        Number of attention heads

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import TriangleAttentionStartingNode
    >>> batch_size, seq_len, c = 2, 10, 32
    >>> n_head = 4
    >>> module = TriangleAttentionStartingNode(c=c, n_head=n_head)
    >>> z_ij = torch.randn(batch_size, seq_len, seq_len, c)
    >>> z_tilde_ij = module(z_ij)
    >>> z_tilde_ij.shape
    torch.Size([2, 10, 10, 32])

    References
    ----------
    .. [1] AlphaFold 3 paper, Algorithm 14: Triangular gated self-attention around starting node
    """

    def __init__(self, c: int = 32, n_head: int = 4):
        super().__init__()

        self.c = c
        self.n_head = n_head
        self.head_dim = c // n_head

        if c % n_head != 0:
            raise ValueError(
                f"Channel dimension {c} must be divisible by number of heads {n_head}"
            )

        # Layer normalization for input (step 1)
        self.layer_norm = nn.LayerNorm(c)

        # Linear projections for queries, keys, values (step 2)
        self.linear_q = nn.Linear(c, c, bias=False)
        self.linear_k = nn.Linear(c, c, bias=False)
        self.linear_v = nn.Linear(c, c, bias=False)

        # Bias projection (step 3)
        self.linear_b = nn.Linear(c, n_head, bias=False)

        # Gate projection (step 4)
        self.linear_g = nn.Linear(c, c, bias=False)

        # Output projection (step 7)
        self.output_linear = nn.Linear(c, c, bias=False)

        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, z_ij: Tensor) -> Tensor:
        r"""
        Forward pass of triangular gated self-attention around starting node.

        Parameters
        ----------
        z_ij : Tensor, shape=(..., s, s, c)
            Input pair representation where s is sequence length
            and c is channel dimension.

        Returns
        -------
        z_tilde_ij : Tensor, shape=(..., s, s, c)
            Updated pair representation after triangular attention.
        """
        batch_shape = z_ij.shape[:-3]
        seq_len = z_ij.shape[-2]

        # Step 1: Layer normalization
        z_ij = self.layer_norm(z_ij)

        # Step 2: Input projections for queries, keys, values
        q_ij = self.linear_q(z_ij)  # (..., s, s, c)
        k_ij = self.linear_k(z_ij)  # (..., s, s, c)
        v_ij = self.linear_v(z_ij)  # (..., s, s, c)

        # Reshape for multi-head attention
        # (..., s, s, n_head, head_dim)
        q_ij = q_ij.view(*batch_shape, seq_len, seq_len, self.n_head, self.head_dim)
        k_ij = k_ij.view(*batch_shape, seq_len, seq_len, self.n_head, self.head_dim)
        v_ij = v_ij.view(*batch_shape, seq_len, seq_len, self.n_head, self.head_dim)

        # Step 3: Bias projection
        b_ij = self.linear_b(z_ij)  # (..., s, s, n_head)

        # Step 4: Gate projection
        g_ij = torch.sigmoid(self.linear_g(z_ij))  # (..., s, s, c)

        # Step 5: Attention computation
        # α^h_ijk = softmax_k(1/√c * q^h_ij^T k^h_jk + b^h_ij)
        # For starting node attention, we attend from j to k for each edge (i,j)

        # Simplify the attention computation using einsum for clarity
        # q_ij: (..., s, s, n_head, head_dim) - query for edge (i,j)
        # k_ij: (..., s, s, n_head, head_dim) - key for edge (i,j)
        # We need k_jk: key for edge (j,k) - this means we need k[j,k] from k[i,j]

        # For each position (i,j), we want to attend to all k positions using k[j,k]
        # This means: for edge (i,j), use query q[i,j] and attend over k[j,:] (all edges starting from j)

        # Compute attention scores using einsum
        # q_ij: (..., i, j, h, d), k_ij: (..., j, k, h, d) -> (..., i, j, h, k)
        attn_scores = torch.einsum("...ijhd,...jkhd->...ijhk", q_ij, k_ij) * self.scale

        # Add bias: b_ij has shape (..., s, s, n_head), we want (..., i, j, h, k)
        # We need to broadcast b[i,j,h] to all k positions
        b_expanded = torch.unsqueeze(b_ij, -1)  # (..., s, s, n_head, 1)
        attn_scores = attn_scores + b_expanded  # (..., s, s, n_head, s)

        # Apply softmax over k dimension (last dimension)
        alpha_ijk = torch.softmax(attn_scores, dim=-1)  # (..., s, s, n_head, s)

        # Step 6: Weighted sum of values
        # o^h_ij = g^h_ij ⊙ ∑_k α^h_ijk v^h_jk

        # Compute weighted sum using einsum
        # alpha_ijk: (..., i, j, h, k), v_ij: (..., j, k, h, d) -> (..., i, j, h, d)
        o_ij = torch.einsum("...ijhk,...jkhd->...ijhd", alpha_ijk, v_ij)

        # Apply gating
        g_expanded = g_ij.view(
            *batch_shape, seq_len, seq_len, self.n_head, self.head_dim
        )
        o_ij = g_expanded * o_ij  # (..., s, s, n_head, head_dim)

        # Step 7: Output projection
        # Concatenate heads
        o_ij_concat = o_ij.view(
            *batch_shape, seq_len, seq_len, self.c
        )  # (..., s, s, c)

        # Final linear projection
        z_tilde_ij = self.output_linear(o_ij_concat)  # (..., s, s, c)

        return z_tilde_ij
