import torch

import beignet


class TimeComputeAlignmentError:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        torch.manual_seed(42)

        device = torch.device("cpu")

        # Generate predicted and target coordinates
        self.predicted_coordinates = torch.randn(
            batch_size, 3, dtype=dtype, device=device
        )
        self.target_coordinates = torch.randn(batch_size, 3, dtype=dtype, device=device)

        # Generate frame definitions - make them well-conditioned
        self.predicted_frames = torch.randn(
            batch_size, 3, 3, 3, dtype=dtype, device=device
        )
        self.target_frames = torch.randn(
            batch_size, 3, 3, 3, dtype=dtype, device=device
        )

        # Make frames non-degenerate by ensuring proper triangular arrangements
        for i in range(3):
            self.predicted_frames[:, i, 0, :] = (
                torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
                + torch.randn(batch_size, 3, dtype=dtype) * 0.1
            )
            self.predicted_frames[:, i, 1, :] = (
                torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
                + torch.randn(batch_size, 3, dtype=dtype) * 0.1
            )
            self.predicted_frames[:, i, 2, :] = (
                torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
                + torch.randn(batch_size, 3, dtype=dtype) * 0.1
            )

            self.target_frames[:, i, 0, :] = (
                torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
                + torch.randn(batch_size, 3, dtype=dtype) * 0.1
            )
            self.target_frames[:, i, 1, :] = (
                torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
                + torch.randn(batch_size, 3, dtype=dtype) * 0.1
            )
            self.target_frames[:, i, 2, :] = (
                torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
                + torch.randn(batch_size, 3, dtype=dtype) * 0.1
            )

        # Compile the function for optimal performance
        self.compiled_compute_alignment_error = torch.compile(
            beignet.compute_alignment_error, fullgraph=True
        )

    def time_compute_alignment_error(self, batch_size, dtype):
        return beignet.compute_alignment_error(
            self.predicted_coordinates,
            self.target_coordinates,
            self.predicted_frames,
            self.target_frames,
        )

    def time_compute_alignment_error_compiled(self, batch_size, dtype):
        return self.compiled_compute_alignment_error(
            self.predicted_coordinates,
            self.target_coordinates,
            self.predicted_frames,
            self.target_frames,
        )


class PeakMemoryComputeAlignmentError:
    params = ([1, 10], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        torch.manual_seed(42)

        device = torch.device("cpu")

        # Generate predicted and target coordinates
        self.predicted_coordinates = torch.randn(
            batch_size, 3, dtype=dtype, device=device
        )
        self.target_coordinates = torch.randn(batch_size, 3, dtype=dtype, device=device)

        # Generate frame definitions - make them well-conditioned
        self.predicted_frames = torch.randn(
            batch_size, 3, 3, 3, dtype=dtype, device=device
        )
        self.target_frames = torch.randn(
            batch_size, 3, 3, 3, dtype=dtype, device=device
        )

        # Make frames non-degenerate
        for i in range(3):
            self.predicted_frames[:, i, 0, :] = (
                torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
                + torch.randn(batch_size, 3, dtype=dtype) * 0.1
            )
            self.predicted_frames[:, i, 1, :] = (
                torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
                + torch.randn(batch_size, 3, dtype=dtype) * 0.1
            )
            self.predicted_frames[:, i, 2, :] = (
                torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
                + torch.randn(batch_size, 3, dtype=dtype) * 0.1
            )

            self.target_frames[:, i, 0, :] = (
                torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
                + torch.randn(batch_size, 3, dtype=dtype) * 0.1
            )
            self.target_frames[:, i, 1, :] = (
                torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
                + torch.randn(batch_size, 3, dtype=dtype) * 0.1
            )
            self.target_frames[:, i, 2, :] = (
                torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
                + torch.randn(batch_size, 3, dtype=dtype) * 0.1
            )

    def peakmem_compute_alignment_error(self, batch_size, dtype):
        return beignet.compute_alignment_error(
            self.predicted_coordinates,
            self.target_coordinates,
            self.predicted_frames,
            self.target_frames,
        )
