import torch

import beignet

from ._set_seed import set_seed


class BenchFTestPower:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.f_test_power,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.df1_values = (
            torch.tensor([2.0, 5.0, 10.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.df2_values = (
            torch.tensor([20.0, 50.0, 100.0], dtype=dtype)
            .repeat(batch_size, 1)
            .flatten()
        )

    def time_f_test_power(self, batch_size, dtype):
        self.func(
            self.effect_sizes,
            self.df1_values,
            self.df2_values,
            alpha=0.05,
        )

    def peak_memory_f_test_power(self, batch_size, dtype):
        self.func(
            self.effect_sizes,
            self.df1_values,
            self.df2_values,
            alpha=0.05,
        )
