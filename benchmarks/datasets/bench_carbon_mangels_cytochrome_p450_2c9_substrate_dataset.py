import os

from beignet.datasets import CarbonMangelsCytochromeP4502C9SubstrateDataset

from .._set_seed import set_seed


class BenchCarbonMangelsCytochromeP4502C9SubstrateDataset:
    params = [
        [10, 100, 1000],
    ]

    param_names = ["batch_size"]

    def setup(self, batch_size):
        set_seed()

        self.dataset = CarbonMangelsCytochromeP4502C9SubstrateDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "."),
            download=False,
            transform=None,
            target_transform=None,
        )

    def time___init__(self, batch_size):
        CarbonMangelsCytochromeP4502C9SubstrateDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "."),
            download=False,
            transform=None,
            target_transform=None,
        )

    def peak_memory___init__(self, batch_size):
        CarbonMangelsCytochromeP4502C9SubstrateDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", ".")
            + "/carbon_mangels_cytochrome_p450_2c9_substrate",
            download=False,
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
