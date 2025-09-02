import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module


class TriangleMultiplicationIncoming(Module):
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
        self.layer_norm = LayerNorm(c)

        # Linear projections without bias for a and b (step 2)
        self.linear_a = Linear(c, c, bias=False)
        self.linear_b = Linear(c, c, bias=False)

        # Linear projection without bias for g (step 3)
        self.linear_g = Linear(c, c, bias=False)

        # Final linear projection without bias with layer norm (step 4)
        self.final_layer_norm = LayerNorm(c)
        self.final_linear = Linear(c, c, bias=False)

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
        z_ij = self.layer_norm(z_ij)

        return torch.sigmoid(self.linear_g(z_ij)) * self.final_linear(
            self.final_layer_norm(
                torch.einsum(
                    "...kic,...kjc->...ijc",
                    torch.sigmoid(self.linear_a(z_ij)).transpose(-3, -2),
                    torch.sigmoid(self.linear_b(z_ij)),
                )
            )
        )
