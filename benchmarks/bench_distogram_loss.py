import os

import torch

import beignet.nn.functional as F

# Set benchmark seed for reproducibility
SEED = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
torch.manual_seed(SEED)


class TimeDistogramLoss:
    """Benchmark distogram loss computation time."""

    params = (
        [1, 4, 16],  # batch_size
        [50, 100, 200],  # n_residues
        [32, 64, 128],  # n_bins
        [torch.float32, torch.float64],  # dtype
    )
    param_names = ["batch_size", "n_residues", "n_bins", "dtype"]

    def setup(self, batch_size, n_residues, n_bins, dtype):
        """Setup test data."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generate test data
        self.logits = torch.randn(
            batch_size, n_residues, n_residues, n_bins, dtype=dtype, device=device
        )

        # Generate symmetric target distances
        target_distances = torch.rand(
            batch_size, n_residues, n_residues, dtype=dtype, device=device
        )
        self.target_distances = (
            target_distances + target_distances.transpose(-2, -1)
        ) / 2
        self.target_distances = self.target_distances * 20.0 + 2.0  # Scale to [2, 22]

        # Generate symmetric mask
        mask = torch.rand(batch_size, n_residues, n_residues, device=device) > 0.2
        self.mask = mask & mask.transpose(-2, -1)
        # Mask diagonal
        self.mask = self.mask & ~torch.eye(
            n_residues, dtype=torch.bool, device=device
        ).unsqueeze(0)

        # Pre-compile the function for consistent benchmarking
        self.compiled_loss = torch.compile(F.distogram_loss, fullgraph=True)

        # Warmup
        for _ in range(3):
            _ = self.compiled_loss(
                self.logits,
                self.target_distances,
                self.mask,
                min_bin=2.3125,
                max_bin=21.6875,
                n_bins=n_bins,
                reduction="mean",
            )

    def time_distogram_loss(self, batch_size, n_residues, n_bins, dtype):
        """Time distogram loss computation."""
        loss = self.compiled_loss(
            self.logits,
            self.target_distances,
            self.mask,
            min_bin=2.3125,
            max_bin=21.6875,
            n_bins=n_bins,
            reduction="mean",
        )
        # Ensure synchronization for accurate timing on GPU
        if self.logits.is_cuda:
            torch.cuda.synchronize()
        return loss


class PeakMemoryDistogramLoss:
    """Benchmark peak memory usage of distogram loss."""

    params = (
        [1, 4],  # batch_size
        [50, 100],  # n_residues
        [32, 64],  # n_bins
        [torch.float32, torch.float64],  # dtype
    )
    param_names = ["batch_size", "n_residues", "n_bins", "dtype"]

    def setup(self, batch_size, n_residues, n_bins, dtype):
        """Setup test data."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generate test data
        self.logits = torch.randn(
            batch_size, n_residues, n_residues, n_bins, dtype=dtype, device=device
        )

        # Generate symmetric target distances
        target_distances = torch.rand(
            batch_size, n_residues, n_residues, dtype=dtype, device=device
        )
        self.target_distances = (
            target_distances + target_distances.transpose(-2, -1)
        ) / 2
        self.target_distances = self.target_distances * 20.0 + 2.0

        # Generate symmetric mask
        mask = torch.rand(batch_size, n_residues, n_residues, device=device) > 0.2
        self.mask = mask & mask.transpose(-2, -1)
        self.mask = self.mask & ~torch.eye(
            n_residues, dtype=torch.bool, device=device
        ).unsqueeze(0)

        # Pre-compile the function
        self.compiled_loss = torch.compile(F.distogram_loss, fullgraph=True)

        # Warmup
        _ = self.compiled_loss(
            self.logits,
            self.target_distances,
            self.mask,
            min_bin=2.3125,
            max_bin=21.6875,
            n_bins=n_bins,
            reduction="mean",
        )

    def peak_memory_distogram_loss(self, batch_size, n_residues, n_bins, dtype):
        """Measure peak memory usage of distogram loss."""
        loss = self.compiled_loss(
            self.logits,
            self.target_distances,
            self.mask,
            min_bin=2.3125,
            max_bin=21.6875,
            n_bins=n_bins,
            reduction="mean",
        )
        # Ensure synchronization for accurate measurement on GPU
        if self.logits.is_cuda:
            torch.cuda.synchronize()
        return loss
