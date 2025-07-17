import os

from beignet.datasets import TDCDataset


class BenchTDCDataset:
    params = [
        [10, 100, 1000],
    ]

    param_names = ["batch_size"]

    def setup(self, batch_size):
        self.dataset = TDCDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "."),
            download=False,
            identifier=4259572,
            suffix="tsv",
            checksum="md5:e8e7c5ba675129db0161913ba4871834",
            x_keys=["Drug"],
            y_keys=["Y"],
            transform=None,
            target_transform=None,
        )

    def time___init__(self, batch_size):
        TDCDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "."),
            download=False,
            identifier=4259572,
            suffix="tsv",
            checksum="md5:e8e7c5ba675129db0161913ba4871834",
            x_keys=["Drug"],
            y_keys=["Y"],
            transform=None,
            target_transform=None,
        )

    def peak_memory___init__(self, batch_size):
        TDCDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "."),
            download=False,
            identifier=4259572,
            suffix="tsv",
            checksum="md5:e8e7c5ba675129db0161913ba4871834",
            x_keys=["Drug"],
            y_keys=["Y"],
            transform=None,
            target_transform=None,
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
