import torch

import beignet

from ._set_seed import set_seed


class BenchZTestPower:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.z_test_power,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.sample_sizes = (
            torch.tensor([20, 50, 100], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def time_z_test_power(self, batch_size, dtype):
        self.func(
            self.effect_sizes,
            self.sample_sizes,
            alpha=0.05,
            alternative="two-sided",
        )

    def peak_memory_z_test_power(self, batch_size, dtype):
        self.func(
            self.effect_sizes,
            self.sample_sizes,
            alpha=0.05,
            alternative="two-sided",
        )
