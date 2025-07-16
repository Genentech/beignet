import random

import torch

import beignet


class ApplyEulerAngle:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.apply_euler_angle,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.rotation = torch.randn(batch_size, dtype=dtype)

        self.axes = random.choice(
            ["x", "y", "z", "xy", "xz", "yz", "xyz", "yx", "zx", "zy", "zyx", "yxz"]
        )
        self.degrees = random.choice([True, False])

        self.inverse = random.choice([True, False])

    def time_apply_euler_angle(self, batch_size, dtype):
        self.func(self.input, self.rotation, self.axes, self.degrees, self.inverse)

    def peak_memory_apply_euler_angle(self, batch_size, dtype):
        self.func(self.input, self.rotation, self.axes, self.degrees, self.inverse)
