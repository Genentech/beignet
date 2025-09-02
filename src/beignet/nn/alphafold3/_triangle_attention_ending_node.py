import math

import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module


class TriangleAttentionEndingNode(Module):
    r"""
    Triangular gated self-attention around ending node from AlphaFold 3.

    This implements Algorithm 15 from AlphaFold 3, which performs triangular
    gated self-attention where the attention is computed around the ending
    node of each edge in the triangle. The key differences from Algorithm 14
    are in steps 5 and 6 where k^h_kj and v^h_kj are used instead of k^h_jk and v^h_jk.

    Parameters
    ----------
    c : int, default=32
        Channel dimension for the pair representation
    n_head : int, default=4
        Number of attention heads

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import TriangleAttentionEndingNode
    >>> batch_size, seq_len, c = 2, 10, 32
    >>> n_head = 4
    >>> module = TriangleAttentionEndingNode(c=c, n_head=n_head)
    >>> z_ij = torch.randn(batch_size, seq_len, seq_len, c)
    >>> z_tilde_ij = module(z_ij)
    >>> z_tilde_ij.shape
    torch.Size([2, 10, 10, 32])

    References
    ----------
    .. [1] AlphaFold 3 paper, Algorithm 15: Triangular gated self-attention around ending node
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
        self.layer_norm = LayerNorm(c)

        # Linear projections for queries, keys, values (step 2)
        self.linear_q = Linear(c, c, bias=False)
        self.linear_k = Linear(c, c, bias=False)
        self.linear_v = Linear(c, c, bias=False)

        # Bias projection (step 3)
        self.linear_b = Linear(c, n_head, bias=False)

        # Gate projection (step 4)
        self.linear_g = Linear(c, c, bias=False)

        # Output projection (step 7)
        self.output_linear = Linear(c, c, bias=False)

        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, z_ij: Tensor) -> Tensor:
        r"""
        Forward pass of triangular gated self-attention around ending node.

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
        # Step 1: Layer normalization
        z_ij = self.layer_norm(z_ij)

        return self.output_linear(
            (
                torch.sigmoid(self.linear_g(z_ij)).view(
                    *(z_ij.shape[:-3]),
                    z_ij.shape[-2],
                    z_ij.shape[-2],
                    self.n_head,
                    self.head_dim,
                )
                * torch.einsum(
                    "...ijhk,...kjhd->...ijhd",
                    torch.softmax(
                        (
                            torch.einsum(
                                "...ijhd,...kjhd->...ijhk",
                                self.linear_q(z_ij).view(
                                    *(z_ij.shape[:-3]),
                                    z_ij.shape[-2],
                                    z_ij.shape[-2],
                                    self.n_head,
                                    self.head_dim,
                                ),
                                self.linear_k(z_ij).view(
                                    *(z_ij.shape[:-3]),
                                    z_ij.shape[-2],
                                    z_ij.shape[-2],
                                    self.n_head,
                                    self.head_dim,
                                ),
                            )
                            * self.scale
                            + torch.unsqueeze(self.linear_b(z_ij), -1)
                        ),
                        dim=-1,
                    ),
                    self.linear_v(z_ij).view(
                        *(z_ij.shape[:-3]),
                        z_ij.shape[-2],
                        z_ij.shape[-2],
                        self.n_head,
                        self.head_dim,
                    ),
                )
            ).view(
                *(z_ij.shape[:-3]),
                z_ij.shape[-2],
                z_ij.shape[-2],
                self.c,
            )
        )
