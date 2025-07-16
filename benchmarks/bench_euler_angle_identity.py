import random

import torch

import beignet


class EulerAngleIdentity:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.euler_angle_identity,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.size = 10

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

        self.degrees = torch.randn(batch_size, dtype=dtype)

        self.out = random.choice([None, torch.randn(batch_size, dtype=dtype)])

        self.dtype = torch.randn(batch_size, dtype=dtype)

        self.layout = torch.randn(batch_size, dtype=dtype)

        self.device = torch.randn(batch_size, dtype=dtype)

        self.requires_grad = torch.randn(batch_size, dtype=dtype)

    def time_euler_angle_identity(self, batch_size, dtype):
        self.func(
            self.input,
            self.size,
            self.axes,
            self.degrees,
            self.out,
            self.dtype,
            self.layout,
            self.device,
            self.requires_grad,
        )

    def peak_memory_euler_angle_identity(self, batch_size, dtype):
        self.func(
            self.input,
            self.size,
            self.axes,
            self.degrees,
            self.out,
            self.dtype,
            self.layout,
            self.device,
            self.requires_grad,
        )
