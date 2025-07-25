import os

from beignet.datasets import ATOM3DDataset

from .._set_seed import set_seed


class BenchATOM3DDataset:
    params = [
        [10, 100, 1000],
    ]

    param_names = ["batch_size"]

    def setup(self, batch_size):
        set_seed()

        self.dataset = ATOM3DDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "."),
            path=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "."),
            resource="https://example.com/atom3d.tar.gz",
            name="atom3d",
            download=False,
            transform=None,
        )

    def time___init__(self, batch_size):
        ATOM3DDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "."),
            path=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "."),
            resource="https://example.com/atom3d.tar.gz",
            name="atom3d",
            download=False,
            transform=None,
        )

    def peak_memory___init__(self, batch_size):
        ATOM3DDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "."),
            path=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "."),
            resource="https://example.com/atom3d.tar.gz",
            name="atom3d",
            download=False,
            transform=None,
        )

    def time___len__(self, batch_size):
        len(self.dataset)

    def peak_memory___len__(self, batch_size):
        len(self.dataset)

    def time___getitem__(self, batch_size):
        for i in range(min(batch_size, len(self.dataset))):
            self.dataset[i]

    def peak_memory___getitem__(self, batch_size):
        for i in range(min(batch_size, len(self.dataset))):
            self.dataset[i]
