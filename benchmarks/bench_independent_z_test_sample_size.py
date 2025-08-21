import torch

import beignet

from ._set_seed import set_seed


class BenchIndependentZTestSampleSize:
    params = [
        [1, 10, 100],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.independent_z_test_sample_size,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.effect_sizes = (
            torch.tensor([0.2, 0.5, 0.8], dtype=dtype).repeat(batch_size, 1).flatten()
        )
        self.ratio_values = (
            torch.tensor([1.0, 1.5, 2.0], dtype=dtype).repeat(batch_size, 1).flatten()
        )

    def time_independent_z_test_sample_size(self, batch_size, dtype):
        self.func(
            self.effect_sizes,
            self.ratio_values,
            power=0.8,
            alpha=0.05,
            alternative="two-sided",
        )

    def peak_memory_independent_z_test_sample_size(self, batch_size, dtype):
        self.func(
            self.effect_sizes,
            self.ratio_values,
            power=0.8,
            alpha=0.05,
            alternative="two-sided",
        )
