import random

import torch

import beignet
from benchmarks._set_seed import set_seed


class BenchRandomEulerAngle:
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
        set_seed()

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

        self.degrees = random.choice([True, False])

        self.generator = random.choice([None, torch.Generator()])

        self.out = random.choice([None, torch.empty((self.size, 3), dtype=dtype)])

        self.dtype = dtype

        self.layout = torch.strided

        self.device = torch.device("cpu")

        self.requires_grad = random.choice([True, False])

        self.pin_memory = random.choice([True, False])

    def time_random_euler_angle(self, batch_size, dtype):
        self.func(
            self.size,
            self.axes,
            self.degrees,
            generator=self.generator,
            out=self.out,
            dtype=self.dtype,
            layout=self.layout,
            device=self.device,
            requires_grad=self.requires_grad,
            pin_memory=self.pin_memory,
        )

    def peak_memory_random_euler_angle(self, batch_size, dtype):
        self.func(
            self.size,
            self.axes,
            self.degrees,
            generator=self.generator,
            out=self.out,
            dtype=self.dtype,
            layout=self.layout,
            device=self.device,
            requires_grad=self.requires_grad,
            pin_memory=self.pin_memory,
        )
