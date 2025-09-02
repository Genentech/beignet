import torch
import torch.nn as nn
from torch import Tensor


class AlphaFold3Distogram(nn.Module):
    r"""
    Distogram Head for AlphaFold 3.

    This module predicts distance distributions (distograms) between pairs of
    residues from pair representations. It outputs probability distributions
    over distance bins, which are useful for structure prediction and validation.

    Parameters
    ----------
    c_z : int, default=128
        Pair representation dimension
    n_bins : int, default=64
        Number of distance bins
    min_dist : float, default=2.3125
        Minimum distance in Angstroms
    max_dist : float, default=21.6875
        Maximum distance in Angstroms

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import AlphaFold3Distogram
    >>> batch_size, n_tokens = 2, 64
    >>> module = AlphaFold3Distogram()
    >>> z_ij = torch.randn(batch_size, n_tokens, n_tokens, 128)
    >>> p_distogram = module(z_ij)
    >>> p_distogram.shape
    torch.Size([2, 64, 64, 64])
    """

    def __init__(
        self,
        c_z: int = 128,
        n_bins: int = 64,
        min_dist: float = 2.3125,
        max_dist: float = 21.6875,
    ):
        super().__init__()

        self.c_z = c_z
        self.n_bins = n_bins
        self.min_dist = min_dist
        self.max_dist = max_dist

        # Create distance bins
        self.register_buffer(
            "distance_bins", torch.linspace(min_dist, max_dist, n_bins)
        )

        # Distogram prediction head
        self.distogram_head = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z, bias=True),
            nn.ReLU(),
            nn.Linear(c_z, n_bins, bias=True),
        )

    def forward(self, z_ij: Tensor) -> Tensor:
        r"""
        Forward pass of Distogram Head.

        Parameters
        ----------
        z_ij : Tensor, shape=(batch_size, n_tokens, n_tokens, c_z)
            Pair representations

        Returns
        -------
        p_distogram : Tensor, shape=(batch_size, n_tokens, n_tokens, n_bins)
            Distance probability distributions (softmax over bins)
        """
        # Apply distogram head
        logits = self.distogram_head(z_ij)  # (batch, n_tokens, n_tokens, n_bins)

        # Apply softmax to get probabilities
        p_distogram = torch.softmax(logits, dim=-1)

        return p_distogram
