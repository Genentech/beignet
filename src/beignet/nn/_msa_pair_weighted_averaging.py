import torch
import torch.nn as nn
from torch import Tensor


class MSAPairWeightedAveraging(nn.Module):
    r"""
    MSA pair weighted averaging with gating from AlphaFold 3.

    This implements Algorithm 10 from AlphaFold 3, which performs weighted
    averaging of MSA representations using pair representations as weights,
    with gating for controlled information flow.

    Parameters
    ----------
    c_m : int, default=32
        Channel dimension for the MSA representation
    c_z : int, default=128
        Channel dimension for the pair representation
    n_head : int, default=8
        Number of attention heads

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import MSAPairWeightedAveraging
    >>> batch_size, seq_len, n_seq, c_m, c_z = 2, 10, 5, 32, 128
    >>> n_head = 8
    >>> module = MSAPairWeightedAveraging(c_m=c_m, c_z=c_z, n_head=n_head)
    >>> m_si = torch.randn(batch_size, seq_len, n_seq, c_m)
    >>> z_ij = torch.randn(batch_size, seq_len, seq_len, c_z)
    >>> m_tilde_si = module(m_si, z_ij)
    >>> m_tilde_si.shape
    torch.Size([2, 10, 5, 32])

    References
    ----------
    .. [1] AlphaFold 3 paper, Algorithm 10: MSA pair weighted averaging with gating
    """

    def __init__(self, c_m: int = 32, c_z: int = 128, n_head: int = 8):
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.n_head = n_head
        self.head_dim = c_m // n_head

        if c_m % n_head != 0:
            raise ValueError(
                f"MSA channel dimension {c_m} must be divisible by number of heads {n_head}"
            )

        # Layer normalization for MSA input (step 1)
        self.msa_layer_norm = nn.LayerNorm(c_m)

        # Linear projection for MSA values (step 2)
        self.linear_v = nn.Linear(c_m, c_m, bias=False)

        # Linear projection for pair weights (step 3)
        self.linear_b = nn.Linear(c_z, n_head, bias=False)
        self.pair_layer_norm = nn.LayerNorm(c_z)

        # Gate projection for MSA (step 4)
        self.linear_g = nn.Linear(c_m, c_m, bias=False)

        # Output projection (step 7)
        self.output_linear = nn.Linear(c_m, c_m, bias=False)

    def forward(self, m_si: Tensor, z_ij: Tensor) -> Tensor:
        r"""
        Forward pass of MSA pair weighted averaging with gating.

        Parameters
        ----------
        m_si : Tensor, shape=(..., s, n_seq, c_m)
            Input MSA representation where s is sequence length,
            n_seq is number of sequences, and c_m is MSA channel dimension.
        z_ij : Tensor, shape=(..., s, s, c_z)
            Input pair representation where s is sequence length
            and c_z is pair channel dimension.

        Returns
        -------
        m_tilde_si : Tensor, shape=(..., s, n_seq, c_m)
            Updated MSA representation after pair-weighted averaging.
        """
        m_si = self.msa_layer_norm(m_si)

        return self.output_linear(
            (
                torch.sigmoid(self.linear_g(m_si)).view(
                    *(m_si.shape[:-3]),
                    m_si.shape[-3],
                    m_si.shape[-2],
                    self.n_head,
                    self.head_dim,
                )
                * torch.einsum(
                    "...ijh,...jnhd->...inhd",
                    torch.softmax(
                        self.linear_b(self.pair_layer_norm(z_ij)),
                        dim=-2,
                    ),
                    self.linear_v(m_si).view(
                        *(m_si.shape[:-3]),
                        m_si.shape[-3],
                        m_si.shape[-2],
                        self.n_head,
                        self.head_dim,
                    ),
                )
            ).view(
                *(m_si.shape[:-3]),
                m_si.shape[-3],
                m_si.shape[-2],
                self.c_m,
            )
        )
