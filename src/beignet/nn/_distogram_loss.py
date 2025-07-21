import torch
import torch.nn as nn
from torch import Tensor

from .functional import distogram_loss


class DistogramLoss(nn.Module):
    r"""
    Distogram loss module for protein structure prediction.

    The distogram loss is a cross-entropy loss between predicted distance
    distributions and target distances. It discretizes continuous distances
    into bins and treats the problem as a classification task.

    Parameters
    ----------
    min_bin : float, default=2.3125
        Minimum distance for binning (in Angstroms). Default value from AlphaFold,
        representing slightly above the minimum possible Cβ-Cβ distance in proteins.
    max_bin : float, default=21.6875
        Maximum distance for binning (in Angstroms). Default value from AlphaFold,
        capturing most meaningful structural contacts.
    n_bins : int, default=64
        Number of distance bins. With default min/max, gives ~0.3 Å resolution.
    reduction : str, default="mean"
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Examples
    --------
    >>> loss_fn = DistogramLoss(min_bin=2.0, max_bin=22.0, n_bins=64)
    >>> batch_size, n_residues = 2, 50
    >>> logits = torch.randn(batch_size, n_residues, n_residues, 64)
    >>> distances = torch.rand(batch_size, n_residues, n_residues) * 20 + 2
    >>> mask = torch.ones(batch_size, n_residues, n_residues)
    >>> loss = loss_fn(logits, distances, mask)
    >>> loss.shape
    torch.Size([])
    """

    def __init__(
        self,
        min_bin: float = 2.3125,  # Å - slightly above minimum Cβ-Cβ distance
        max_bin: float = 21.6875,  # Å - captures most structural contacts
        n_bins: int = 64,  # gives ~0.3 Å resolution per bin
        reduction: str = "mean",
    ):
        super().__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.n_bins = n_bins
        self.reduction = reduction
        # Precompute bin width for efficiency
        self.register_buffer("bin_width", torch.tensor((max_bin - min_bin) / n_bins))

    def forward(
        self,
        logits: Tensor,
        target_distances: Tensor,
        mask: Tensor,
    ) -> Tensor:
        r"""
        Compute the distogram loss.

        Parameters
        ----------
        logits : Tensor, shape=(..., N, N, n_bins)
            Predicted logits for distance bins. N is the number of residues.
        target_distances : Tensor, shape=(..., N, N)
            True distances between residue pairs.
        mask : Tensor, shape=(..., N, N)
            Binary mask indicating valid residue pairs (1 for valid, 0 for invalid).

        Returns
        -------
        loss : Tensor
            The distogram loss. Shape depends on reduction:
            - 'none': shape=(..., N, N)
            - 'mean' or 'sum': scalar
        """
        return distogram_loss(
            logits,
            target_distances,
            mask,
            min_bin=self.min_bin,
            max_bin=self.max_bin,
            n_bins=self.n_bins,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return (
            f"min_bin={self.min_bin}, max_bin={self.max_bin}, "
            f"n_bins={self.n_bins}, reduction={self.reduction}"
        )
