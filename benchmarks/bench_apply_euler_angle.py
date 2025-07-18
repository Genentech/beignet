import random

import torch

import beignet

from ._set_seed import set_seed


class BenchApplyEulerAngle:
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
        set_seed()

        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.rotation = torch.randn(batch_size, dtype=dtype)

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

        self.inverse = random.choice([True, False])

    def time_apply_euler_angle(self, batch_size, dtype):
        self.func(self.input, self.rotation, self.axes, self.degrees, self.inverse)

    def peak_memory_apply_euler_angle(self, batch_size, dtype):
        self.func(self.input, self.rotation, self.axes, self.degrees, self.inverse)
