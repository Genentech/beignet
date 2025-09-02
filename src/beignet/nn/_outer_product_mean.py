import torch
import torch.nn as nn
from torch import Tensor


class OuterProductMean(nn.Module):
    r"""
    Outer Product Mean module from AlphaFold 3.

    This implements Algorithm 9 from AlphaFold 3, which computes outer products
    between MSA sequence representations and applies linear transformations.
    The "mean" in outer product mean refers to averaging outer products computed
    across all MSA sequences, capturing coevolutionary information.

    The algorithm follows these steps:
    1. Apply layer normalization to input MSA representation
    2. Compute linear projections without bias to get two vectors for each sequence
    3. For each MSA sequence s, compute pairwise outer products a_s,i ⊗ b_s,j
       for all residue pairs (i, j), flatten the (c × c) matrix to length c*c
    4. Average outer products over all MSA sequences (this is the "mean")
    5. Apply final linear transformation to produce pair features

    Parameters
    ----------
    c : int, default=32
        Input channel dimension
    c_z : int, default=128
        Output channel dimension for final projection

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import OuterProductMean
    >>> batch_size, seq_len, n_seq, c = 2, 10, 16, 32
    >>> c_z = 128
    >>> module = OuterProductMean(c=c, c_z=c_z)
    >>> m_si = torch.randn(batch_size, seq_len, n_seq, c)  # MSA representation
    >>> z_ij = module(m_si)
    >>> z_ij.shape
    torch.Size([2, 10, 10, 128])

    References
    ----------
    .. [1] AlphaFold 3 paper, Algorithm 9: Outer product mean
    """

    def __init__(self, c: int = 32, c_z: int = 128):
        super().__init__()

        self.c = c
        self.c_z = c_z

        # Layer normalization (step 1)
        self.layer_norm = nn.LayerNorm(c)

        # Linear projection without bias to get a and b vectors (step 2)
        self.linear_no_bias = nn.Linear(c, 2 * c, bias=False)

        # Final linear projection (step 4)
        self.final_linear = nn.Linear(c * c, c_z)

    def forward(self, m_si: Tensor) -> Tensor:
        r"""
        Forward pass of outer product mean.

        Parameters
        ----------
        m_si : Tensor, shape=(..., s, n_seq, c)
            Input MSA representation where s is sequence length,
            n_seq is number of MSA sequences, and c is channel dimension.

        Returns
        -------
        z_ij : Tensor, shape=(..., s, s, c_z)
            Pairwise representation after outer product mean operation.
        """
        m_si = self.layer_norm(m_si)  # (..., s, n_seq, c)

        a_si, b_si = torch.chunk(
            self.linear_no_bias(m_si),
            2,
            dim=-1,
        )

        return self.final_linear(
            torch.mean(
                torch.flatten(
                    torch.unsqueeze(torch.unsqueeze(a_si, -3), -1)
                    * torch.unsqueeze(torch.unsqueeze(b_si, -4), -2),
                    start_dim=-2,
                ),
                dim=-2,
            )
        )
