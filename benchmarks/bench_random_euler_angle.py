import random

import torch

import beignet


class RandomEulerAngle:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.random_euler_angle,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.size = 10

        self.axes = random.choice(
            ["x", "y", "z", "xy", "xz", "yz", "xyz", "yx", "zx", "zy", "zyx", "yxz"]
        )
        self.degrees = torch.randn(batch_size, dtype=dtype)

        self.generator = random.choice([None, torch.Generator()])

        self.out = random.choice([None, torch.randn(batch_size, dtype=dtype)])

        self.dtype = torch.randn(batch_size, dtype=dtype)

        self.layout = torch.randn(batch_size, dtype=dtype)

        self.device = torch.randn(batch_size, dtype=dtype)

        self.requires_grad = torch.randn(batch_size, dtype=dtype)

        self.pin_memory = torch.randn(batch_size, dtype=dtype)

    def time_random_euler_angle(self, batch_size, dtype):
        self.func(
            self.input,
            self.size,
            self.axes,
            self.degrees,
            self.generator,
            self.out,
            self.dtype,
            self.layout,
            self.device,
            self.requires_grad,
            self.pin_memory,
        )

    def peak_memory_random_euler_angle(self, batch_size, dtype):
        self.func(
            self.input,
            self.size,
            self.axes,
            self.degrees,
            self.generator,
            self.out,
            self.dtype,
            self.layout,
            self.device,
            self.requires_grad,
            self.pin_memory,
        )
