import torch

import beignet


class BenchQuaternionSlerp:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.quaternion_slerp,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.time = torch.randn(batch_size, dtype=dtype)

        self.rotation = torch.randn(batch_size, dtype=dtype)

    def time_quaternion_slerp(self, batch_size, dtype):
        self.func(self.input, self.time, self.rotation)

    def peak_memory_quaternion_slerp(self, batch_size, dtype):
        self.func(self.input, self.time, self.rotation)
