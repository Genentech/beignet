import torch

import beignet


class TimeSmoothLocalDistanceDifferenceTest:
    params = ([1, 10, 100], [16, 64, 128], [torch.float32, torch.float64])
    param_names = ["batch_size", "n_atoms", "dtype"]

    def setup(self, batch_size, n_atoms, dtype):
        # Set random seed for reproducible benchmarks
        torch.manual_seed(42)

        device = torch.device("cpu")

        # Generate predicted atom positions scaled for realistic protein distances
        self.input = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device) * 5

        # Generate target positions (also realistic protein scale)
        self.target = (
            torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device) * 5
        )

        # Make some positions closer together to ensure valid pairs within cutoff
        # This prevents edge cases where no pairs are within the cutoff radius
        for i in range(min(10, n_atoms)):  # First 10 atoms close together
            self.input[:, i] = (
                torch.randn(batch_size, 3, dtype=dtype, device=device) * 2
            )
            self.target[:, i] = (
                torch.randn(batch_size, 3, dtype=dtype, device=device) * 2
            )

        # Compile the function for optimal performance
        self.compiled_smooth_local_distance_difference_test = torch.compile(
            beignet.smooth_local_distance_difference_test, fullgraph=True
        )

    def time_smooth_local_distance_difference_test(self, batch_size, n_atoms, dtype):
        return beignet.smooth_local_distance_difference_test(self.input, self.target)

    def time_smooth_local_distance_difference_test_compiled(
        self, batch_size, n_atoms, dtype
    ):
        return self.compiled_smooth_local_distance_difference_test(
            self.input, self.target
        )

    def time_smooth_local_distance_difference_test_custom_cutoff(
        self, batch_size, n_atoms, dtype
    ):
        return beignet.smooth_local_distance_difference_test(
            self.input, self.target, cutoff_radius=10.0
        )

    def time_smooth_local_distance_difference_test_custom_thresholds(
        self, batch_size, n_atoms, dtype
    ):
        return beignet.smooth_local_distance_difference_test(
            self.input, self.target, tolerance_thresholds=(1.0, 2.0, 4.0)
        )

    def time_smooth_local_distance_difference_test_high_smoothing(
        self, batch_size, n_atoms, dtype
    ):
        return beignet.smooth_local_distance_difference_test(
            self.input, self.target, smoothing_factor=5.0
        )


class PeakMemorySmoothLocalDistanceDifferenceTest:
    params = ([1, 10], [64, 128], [torch.float32])
    param_names = ["batch_size", "n_atoms", "dtype"]

    def setup(self, batch_size, n_atoms, dtype):
        # Set random seed for reproducible benchmarks
        torch.manual_seed(42)

        device = torch.device("cpu")

        # Generate predicted atom positions scaled for realistic protein distances
        self.input = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device) * 5

        # Generate target positions (also realistic protein scale)
        self.target = (
            torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device) * 5
        )

        # Make some positions closer together to ensure valid pairs within cutoff
        for i in range(min(10, n_atoms)):  # First 10 atoms close together
            self.input[:, i] = (
                torch.randn(batch_size, 3, dtype=dtype, device=device) * 2
            )
            self.target[:, i] = (
                torch.randn(batch_size, 3, dtype=dtype, device=device) * 2
            )

    def peakmem_smooth_local_distance_difference_test(self, batch_size, n_atoms, dtype):
        return beignet.smooth_local_distance_difference_test(self.input, self.target)
