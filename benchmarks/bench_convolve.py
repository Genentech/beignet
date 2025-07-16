import torch

import beignet


class Convolve:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.convolve,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.other = torch.randn(batch_size, dtype=dtype)

        self.mode = torch.randn(batch_size, dtype=dtype)

    def time_convolve(self, batch_size, dtype):
        self.func(self.input, self.other, self.mode)

    def peak_memory_convolve(self, batch_size, dtype):
        self.func(self.input, self.other, self.mode)
