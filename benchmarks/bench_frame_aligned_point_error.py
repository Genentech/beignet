import torch

import beignet


class TimeFrameAlignedPointError:
    params = ([1, 10, 100], [1, 4, 16], [10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "n_frames", "n_atoms", "dtype"]

    def setup(self, batch_size, n_frames, n_atoms, dtype):
        # Set random seed for reproducible benchmarks
        torch.manual_seed(42)

        device = torch.device("cpu")

        # Generate orthogonal rotation matrices
        frame_rot = torch.randn(batch_size, n_frames, 3, 3, dtype=dtype, device=device)
        self.frame_rot = torch.linalg.qr(frame_rot).Q
        self.frame_trans = torch.randn(
            batch_size, n_frames, 3, dtype=dtype, device=device
        )

        target_rot = torch.randn(batch_size, n_frames, 3, 3, dtype=dtype, device=device)
        self.target_rot = torch.linalg.qr(target_rot).Q
        self.target_trans = torch.randn(
            batch_size, n_frames, 3, dtype=dtype, device=device
        )

        # Generate atom positions
        self.pos = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.target_pos = torch.randn(
            batch_size, n_atoms, 3, dtype=dtype, device=device
        )

        # Compile the function for optimal performance
        self.compiled_frame_aligned_point_error = torch.compile(
            beignet.frame_aligned_point_error, fullgraph=True
        )

    def time_frame_aligned_point_error(self, batch_size, n_frames, n_atoms, dtype):
        return beignet.frame_aligned_point_error(
            self.frame_rot,
            self.frame_trans,
            self.pos,
            self.target_rot,
            self.target_trans,
            self.target_pos,
        )

    def time_frame_aligned_point_error_compiled(
        self, batch_size, n_frames, n_atoms, dtype
    ):
        return self.compiled_frame_aligned_point_error(
            self.frame_rot,
            self.frame_trans,
            self.pos,
            self.target_rot,
            self.target_trans,
            self.target_pos,
        )

    def time_frame_aligned_point_error_unclamped(
        self, batch_size, n_frames, n_atoms, dtype
    ):
        return beignet.frame_aligned_point_error(
            self.frame_rot,
            self.frame_trans,
            self.pos,
            self.target_rot,
            self.target_trans,
            self.target_pos,
            clamp_distance=None,
        )


class PeakMemoryFrameAlignedPointError:
    params = ([1, 10], [4, 16], [100, 1000], [torch.float32])
    param_names = ["batch_size", "n_frames", "n_atoms", "dtype"]

    def setup(self, batch_size, n_frames, n_atoms, dtype):
        # Set random seed for reproducible benchmarks
        torch.manual_seed(42)

        device = torch.device("cpu")

        # Generate orthogonal rotation matrices
        frame_rot = torch.randn(batch_size, n_frames, 3, 3, dtype=dtype, device=device)
        self.frame_rot = torch.linalg.qr(frame_rot).Q
        self.frame_trans = torch.randn(
            batch_size, n_frames, 3, dtype=dtype, device=device
        )

        target_rot = torch.randn(batch_size, n_frames, 3, 3, dtype=dtype, device=device)
        self.target_rot = torch.linalg.qr(target_rot).Q
        self.target_trans = torch.randn(
            batch_size, n_frames, 3, dtype=dtype, device=device
        )

        # Generate atom positions
        self.pos = torch.randn(batch_size, n_atoms, 3, dtype=dtype, device=device)
        self.target_pos = torch.randn(
            batch_size, n_atoms, 3, dtype=dtype, device=device
        )

    def peakmem_frame_aligned_point_error(self, batch_size, n_frames, n_atoms, dtype):
        return beignet.frame_aligned_point_error(
            self.frame_rot,
            self.frame_trans,
            self.pos,
            self.target_rot,
            self.target_trans,
            self.target_pos,
        )
