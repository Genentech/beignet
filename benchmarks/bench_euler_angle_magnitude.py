import random

import torch

import beignet

from ._set_seed import set_seed


class BenchEulerAngleMagnitude:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.euler_angle_magnitude,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.axes = random.choice(
            [
                "xyz",
                "xzy",
                "yxz",
                "yzx",
                "zxy",
                "zyx",
            ]
        )

        self.degrees = random.choice([True, False])

    def time_euler_angle_magnitude(self, batch_size, dtype):
        self.func(self.input, self.axes, self.degrees)

    def peak_memory_euler_angle_magnitude(self, batch_size, dtype):
        self.func(self.input, self.axes, self.degrees)
