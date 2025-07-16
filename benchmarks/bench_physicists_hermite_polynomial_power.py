import torch

import beignet


class PhysicistsHermitePolynomialPower:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.physicists_hermite_polynomial_power,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.exponent = torch.randn(batch_size, dtype=dtype)

        self.maximum_exponent = torch.randn(batch_size, dtype=dtype)

    def time_physicists_hermite_polynomial_power(self, batch_size, dtype):
        self.func(self.input, self.exponent, self.maximum_exponent)

    def peak_memory_physicists_hermite_polynomial_power(self, batch_size, dtype):
        self.func(self.input, self.exponent, self.maximum_exponent)
