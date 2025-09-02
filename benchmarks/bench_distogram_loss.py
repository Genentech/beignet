import torch

import beignet


class TimeDistogramLoss:
    params = ([1, 10, 100], [16, 64, 128], [32, 64], [torch.float32, torch.float64])
    param_names = ["batch_size", "n_atoms", "num_bins", "dtype"]

    def setup(self, batch_size, n_atoms, num_bins, dtype):
        # Set random seed for reproducible benchmarks
        torch.manual_seed(42)

        device = torch.device("cpu")

        # Generate predicted logits (symmetric for realistic case)
        input_logits = torch.randn(
            batch_size, n_atoms, n_atoms, num_bins, dtype=dtype, device=device
        )
        # Make logits symmetric for more realistic test
        self.input = (input_logits + input_logits.transpose(-3, -2)) / 2

        # Generate random atom positions scaled for realistic distances
        self.target = (
            torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device) * 10
        )

        # Store parameters for use in benchmark methods
        self.num_bins = num_bins

        # Compile the function for optimal performance
        self.compiled_distogram_loss = torch.compile(
            beignet.distogram_loss, fullgraph=True
        )

    def time_distogram_loss(self, batch_size, n_atoms, num_bins, dtype):
        return beignet.distogram_loss(self.input, self.target, num_bins=self.num_bins)

    def time_distogram_loss_compiled(self, batch_size, n_atoms, num_bins, dtype):
        return self.compiled_distogram_loss(
            self.input, self.target, num_bins=self.num_bins
        )

    def time_distogram_loss_custom_range(self, batch_size, n_atoms, num_bins, dtype):
        return beignet.distogram_loss(
            self.input,
            self.target,
            min_distance=1.0,
            max_distance=30.0,
            num_bins=self.num_bins,
        )


class PeakMemoryDistogramLoss:
    params = ([1, 10], [64, 128], [64], [torch.float32])
    param_names = ["batch_size", "n_atoms", "num_bins", "dtype"]

    def setup(self, batch_size, n_atoms, num_bins, dtype):
        # Set random seed for reproducible benchmarks
        torch.manual_seed(42)

        device = torch.device("cpu")

        # Generate predicted logits (symmetric for realistic case)
        input_logits = torch.randn(
            batch_size, n_atoms, n_atoms, num_bins, dtype=dtype, device=device
        )
        # Make logits symmetric for more realistic test
        self.input = (input_logits + input_logits.transpose(-3, -2)) / 2

        # Generate random atom positions scaled for realistic distances
        self.target = (
            torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device) * 10
        )

        # Store parameters
        self.num_bins = num_bins

    def peakmem_distogram_loss(self, batch_size, n_atoms, num_bins, dtype):
        return beignet.distogram_loss(self.input, self.target, num_bins=self.num_bins)
