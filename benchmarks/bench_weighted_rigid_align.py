import torch

import beignet


class TimeWeightedRigidAlign:
    params = ([1, 10, 100], [16, 64, 128, 256], [torch.float32, torch.float64])
    param_names = ["batch_size", "n_points", "dtype"]

    def setup(self, batch_size, n_points, dtype):
        # Set random seed for reproducible benchmarks
        torch.manual_seed(42)

        device = torch.device("cpu")

        # Generate random 3D points for realistic geometric data
        self.input = torch.randn(batch_size, n_points, 3, dtype=dtype, device=device)
        self.target = torch.randn(batch_size, n_points, 3, dtype=dtype, device=device)

        # Generate random positive weights
        self.weights = (
            torch.rand(batch_size, n_points, dtype=dtype, device=device) + 0.1
        )

        # Compile the function for optimal performance
        self.compiled_weighted_rigid_align = torch.compile(
            beignet.weighted_rigid_align, fullgraph=True
        )

    def time_weighted_rigid_align(self, batch_size, n_points, dtype):
        return beignet.weighted_rigid_align(self.input, self.target, self.weights)

    def time_weighted_rigid_align_compiled(self, batch_size, n_points, dtype):
        return self.compiled_weighted_rigid_align(self.input, self.target, self.weights)

    def time_weighted_rigid_align_uniform_weights(self, batch_size, n_points, dtype):
        # Test with uniform weights (equivalent to standard Kabsch)
        uniform_weights = torch.ones_like(self.weights)
        return beignet.weighted_rigid_align(self.input, self.target, uniform_weights)

    def time_weighted_rigid_align_sparse_weights(self, batch_size, n_points, dtype):
        # Test with sparse weights (some zeros)
        sparse_weights = self.weights.clone()
        sparse_weights[:, : n_points // 2] = 0.0  # Zero out first half
        return beignet.weighted_rigid_align(self.input, self.target, sparse_weights)


class PeakMemoryWeightedRigidAlign:
    params = ([1, 10], [128, 256], [torch.float32])
    param_names = ["batch_size", "n_points", "dtype"]

    def setup(self, batch_size, n_points, dtype):
        # Set random seed for reproducible benchmarks
        torch.manual_seed(42)

        device = torch.device("cpu")

        # Generate random 3D points for realistic geometric data
        self.input = torch.randn(batch_size, n_points, 3, dtype=dtype, device=device)
        self.target = torch.randn(batch_size, n_points, 3, dtype=dtype, device=device)

        # Generate random positive weights
        self.weights = (
            torch.rand(batch_size, n_points, dtype=dtype, device=device) + 0.1
        )

    def peakmem_weighted_rigid_align(self, batch_size, n_points, dtype):
        return beignet.weighted_rigid_align(self.input, self.target, self.weights)
