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
        # Step 1: Layer normalization applied to each MSA sequence
        m_si = self.layer_norm(m_si)  # (..., s, n_seq, c)

        # Step 2: Linear projection without bias to get a and b for each MSA sequence
        ab = self.linear_no_bias(m_si)  # (..., s, n_seq, 2*c)
        a_si, b_si = torch.chunk(ab, 2, dim=-1)  # Each: (..., s, n_seq, c)

        # Step 3: Compute outer products for all residue pairs across all MSA sequences
        # For each MSA sequence, compute outer products between all residue pairs

        # Expand dimensions for pairwise outer products
        # a_si: (..., s, n_seq, c) -> (..., s, 1, n_seq, c) (expand for j dimension)
        # b_si: (..., s, n_seq, c) -> (..., 1, s, n_seq, c) (expand for i dimension)
        a_expanded = torch.unsqueeze(a_si, -3)  # (..., s, 1, n_seq, c)
        b_expanded = torch.unsqueeze(b_si, -4)  # (..., 1, s, n_seq, c)

        # Compute outer product: (..., s, s, n_seq, c, c)
        # For each MSA sequence seq_idx, compute outer product a[i, seq_idx] ⊗ b[j, seq_idx]
        outer_product = torch.unsqueeze(a_expanded, -1) * torch.unsqueeze(
            b_expanded, -2
        )

        # Flatten the c×c outer product: (..., s, s, n_seq, c*c)
        outer_product_flat = torch.flatten(outer_product, start_dim=-2)

        # Step 4: Average over MSA sequences (this is the "mean" in outer product mean)
        # This captures coevolutionary information by averaging across evolutionary sequences
        o_ij_mean = torch.mean(outer_product_flat, dim=-2)  # (..., s, s, c*c)

        # Step 5: Final linear transformation
        z_ij = self.final_linear(o_ij_mean)  # (..., s, s, c_z)

        return z_ij
