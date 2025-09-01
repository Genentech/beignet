import torch

import beignet
from benchmarks._set_seed import set_seed


class BenchRotationMatrixMagnitude:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.rotation_matrix_magnitude,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

    def time_rotation_matrix_magnitude(self, batch_size, dtype):
        self.func(self.input)

    def peak_memory_rotation_matrix_magnitude(self, batch_size, dtype):
        self.func(self.input)
