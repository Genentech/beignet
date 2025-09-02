import torch

import beignet


class TimeExpressCoordinatesInFrame:
    params = ([1, 10, 100], [torch.float32, torch.float64])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        # Set random seed for reproducible benchmarks
        torch.manual_seed(42)

        device = torch.device("cpu")

        # Generate coordinates to transform
        self.coordinates = torch.randn(batch_size, 3, dtype=dtype, device=device)

        # Generate frame atoms (a, b, c) - make them form proper triangles
        self.frames = torch.randn(batch_size, 3, 3, dtype=dtype, device=device)

        # Make frames well-conditioned
        self.frames[:, 0, :] = (
            torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
            + torch.randn(batch_size, 3, dtype=dtype) * 0.1
        )  # a
        self.frames[:, 1, :] = (
            torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
            + torch.randn(batch_size, 3, dtype=dtype) * 0.1
        )  # b (center)
        self.frames[:, 2, :] = (
            torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
            + torch.randn(batch_size, 3, dtype=dtype) * 0.1
        )  # c

        # Compile the function for optimal performance
        self.compiled_express_coordinates = torch.compile(
            beignet.express_coordinates_in_frame, fullgraph=True
        )

    def time_express_coordinates_in_frame(self, batch_size, dtype):
        return beignet.express_coordinates_in_frame(self.coordinates, self.frames)

    def time_express_coordinates_in_frame_compiled(self, batch_size, dtype):
        return self.compiled_express_coordinates(self.coordinates, self.frames)


class PeakMemoryExpressCoordinatesInFrame:
    params = ([1, 10], [torch.float32])
    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        # Set random seed for reproducible benchmarks
        torch.manual_seed(42)

        device = torch.device("cpu")

        # Generate coordinates to transform
        self.coordinates = torch.randn(batch_size, 3, dtype=dtype, device=device)

        # Generate frame atoms (a, b, c) - make them form proper triangles
        self.frames = torch.randn(batch_size, 3, 3, dtype=dtype, device=device)

        # Make frames well-conditioned
        self.frames[:, 0, :] = (
            torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
            + torch.randn(batch_size, 3, dtype=dtype) * 0.1
        )
        self.frames[:, 1, :] = (
            torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
            + torch.randn(batch_size, 3, dtype=dtype) * 0.1
        )
        self.frames[:, 2, :] = (
            torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
            + torch.randn(batch_size, 3, dtype=dtype) * 0.1
        )

    def peakmem_express_coordinates_in_frame(self, batch_size, dtype):
        return beignet.express_coordinates_in_frame(self.coordinates, self.frames)
