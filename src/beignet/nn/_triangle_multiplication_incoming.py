import torch
import torch.nn as nn
from torch import Tensor


class TriangleMultiplicationIncoming(nn.Module):
    r"""
    Triangular multiplicative update using "incoming" edges from AlphaFold 3.

    This implements Algorithm 13 from AlphaFold 3, which performs triangular
    multiplicative updates on pair representations using incoming edges.
    The key difference from the outgoing version is in step 4 where
    a_ki ⊙ b_kj is computed instead of a_ik ⊙ b_jk.

    Parameters
    ----------
    c : int, default=128
        Channel dimension for the pair representation

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import TriangleMultiplicationIncoming
    >>> batch_size, seq_len, c = 2, 10, 128
    >>> module = TriangleMultiplicationIncoming(c=c)
    >>> z_ij = torch.randn(batch_size, seq_len, seq_len, c)
    >>> z_tilde_ij = module(z_ij)
    >>> z_tilde_ij.shape
    torch.Size([2, 10, 10, 128])

    References
    ----------
    .. [1] AlphaFold 3 paper, Algorithm 13: Triangular multiplicative update using "incoming" edges
    """

    def __init__(self, c: int = 128):
        super().__init__()

        self.c = c

        # Layer normalization (step 1)
        self.layer_norm = nn.LayerNorm(c)

        # Linear projections without bias for a and b (step 2)
        self.linear_a = nn.Linear(c, c, bias=False)
        self.linear_b = nn.Linear(c, c, bias=False)

        # Linear projection without bias for g (step 3)
        self.linear_g = nn.Linear(c, c, bias=False)

        # Final linear projection without bias with layer norm (step 4)
        self.final_layer_norm = nn.LayerNorm(c)
        self.final_linear = nn.Linear(c, c, bias=False)

    def forward(self, z_ij: Tensor) -> Tensor:
        r"""
        Forward pass of triangular multiplicative update with incoming edges.

        Parameters
        ----------
        z_ij : Tensor, shape=(..., s, s, c)
            Input pair representation where s is sequence length
            and c is channel dimension.

        Returns
        -------
        z_tilde_ij : Tensor, shape=(..., s, s, c)
            Updated pair representation after triangular multiplicative update.
        """
        # Step 1: Layer normalization
        z_ij = self.layer_norm(z_ij)

        # Step 2: Linear projections for a and b with sigmoid activation
        a_ij = torch.sigmoid(self.linear_a(z_ij))  # (..., s, s, c)
        b_ij = torch.sigmoid(self.linear_b(z_ij))  # (..., s, s, c)

        # Step 3: Linear projection for g with sigmoid activation
        g_ij = torch.sigmoid(self.linear_g(z_ij))  # (..., s, s, c)

        # Step 4: Triangular multiplicative update (incoming)
        # z̃_ij = g_ij ⊙ LinearNoBias(LayerNorm(∑_k a_ki ⊙ b_kj))
        # Use einsum to explicitly reduce over k.
        # a^T: (..., k, i, c), b: (..., k, j, c)
        sum_k = torch.einsum("...kic,...kjc->...ijc", a_ij.transpose(-3, -2), b_ij)

        # Apply layer norm and final linear projection
        normalized = self.final_layer_norm(sum_k)  # (..., s, s, c)
        linear_output = self.final_linear(normalized)  # (..., s, s, c)

        # Apply gating
        z_tilde_ij = g_ij * linear_output  # (..., s, s, c)

        return z_tilde_ij
