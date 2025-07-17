import random

import torch

import beignet


class BenchTranslationIdentity:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.translation_identity,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.size = 10

        self.out = random.choice([None, torch.empty((self.size, 3), dtype=dtype)])

        self.dtype = dtype

        self.layout = torch.strided

        self.device = torch.device("cpu")

        self.requires_grad = random.choice([True, False])

    def time_translation_identity(self, batch_size, dtype):
        self.func(
            self.size,
            out=self.out,
            dtype=self.dtype,
            layout=self.layout,
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def peak_memory_translation_identity(self, batch_size, dtype):
        self.func(
            self.size,
            out=self.out,
            dtype=self.dtype,
            layout=self.layout,
            device=self.device,
            requires_grad=self.requires_grad,
        )
