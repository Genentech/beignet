import torch

import beignet

from ._set_seed import set_seed


class BenchEvaluateProbabilistsHermitePolynomialCartesian2D:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.evaluate_probabilists_hermite_polynomial_cartersian_2d,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.x = torch.randn(batch_size, 3, dtype=dtype)

        self.y = torch.randn(batch_size, dtype=dtype)

        self.c = torch.randn(batch_size, dtype=dtype)

    def time_evaluate_probabilists_hermite_polynomial_cartesian_2d(
        self,
        batch_size,
        dtype,
    ):
        self.func(self.c, self.x, self.y)

    def peak_memory_evaluate_probabilists_hermite_polynomial_cartesian_2d(
        self,
        batch_size,
        dtype,
    ):
        self.func(self.c, self.x, self.y)
