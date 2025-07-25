import random

import torch

import beignet

from ._set_seed import set_seed


class BenchEulerAngleIdentity:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.euler_angle_identity,
            fullgraph=False,
        )

    def setup(self, batch_size, dtype):
        set_seed()

        self.size = batch_size

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

        self.out = random.choice([None, torch.randn(batch_size, 3, dtype=dtype)])

        self.dtype = dtype

        self.layout = torch.strided

        self.device = torch.device("cpu")

        self.requires_grad = random.choice([True, False])

    def time_euler_angle_identity(self, batch_size, dtype):
        self.func(
            self.size,
            self.axes,
            self.degrees,
            out=self.out,
            dtype=self.dtype,
            layout=self.layout,
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def peak_memory_euler_angle_identity(self, batch_size, dtype):
        self.func(
            self.size,
            self.axes,
            self.degrees,
            out=self.out,
            dtype=self.dtype,
            layout=self.layout,
            device=self.device,
            requires_grad=self.requires_grad,
        )
