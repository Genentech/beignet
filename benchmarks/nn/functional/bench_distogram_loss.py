import torch

import beignet.nn.functional

from ..._set_seed import set_seed


class BenchDistogramLoss:
    params = [
        [1, 4, 8],  # batch_size
        [50, 100, 200],  # n_residues
        [32, 64],  # n_bins
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "n_residues", "n_bins", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.nn.functional.distogram_loss,
            fullgraph=True,
        )

    def setup(self, batch_size, n_residues, n_bins, dtype):
        set_seed()

        # Setup input tensors
        self.logits = torch.randn(
            batch_size, n_residues, n_residues, n_bins, dtype=dtype
        )

        # Generate symmetric distances
        distances = torch.rand(batch_size, n_residues, n_residues, dtype=dtype)
        self.target_distances = (distances + distances.transpose(-2, -1)) / 2
        self.target_distances = self.target_distances * 20.0 + 2.0  # Scale to [2, 22]

        # Generate mask (use float for compatibility)
        self.mask = torch.ones(batch_size, n_residues, n_residues, dtype=dtype)

        # Store n_bins for the function call
        self.n_bins = n_bins

    def time_distogram_loss(self, batch_size, n_residues, n_bins, dtype):
        self.func(
            self.logits,
            self.target_distances,
            self.mask,
            min_bin=2.3125,  # AlphaFold default
            max_bin=21.6875,  # AlphaFold default
            n_bins=self.n_bins,
            reduction="mean",
        )

    def peak_memory_distogram_loss(self, batch_size, n_residues, n_bins, dtype):
        self.func(
            self.logits,
            self.target_distances,
            self.mask,
            min_bin=2.3125,  # AlphaFold default
            max_bin=21.6875,  # AlphaFold default
            n_bins=self.n_bins,
            reduction="mean",
        )
