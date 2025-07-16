import torch

import beignet


class ApplyRotationMatrixBenchmark:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.apply_rotation_matrix,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.rotation = torch.randn(batch_size, dtype=dtype)

        self.inverse = torch.randn(batch_size, dtype=dtype)

    def time_apply_rotation_matrix(self, batch_size, dtype):
        self.func(self.input, self.rotation, self.inverse)

    def peak_memory_apply_rotation_matrix(self, batch_size, dtype):
        self.func(self.input, self.rotation, self.inverse)
