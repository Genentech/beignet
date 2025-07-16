import os

from beignet.datasets import VeithCytochromeP4503A4InhibitionDataset


class BenchVeithCytochromeP4503A4InhibitionDataset:
    params = [
        [10, 100, 1000],
    ]

    param_names = ["batch_size"]

    def __init__(self):
        pass

    def setup(self, batch_size):
        self.dataset = VeithCytochromeP4503A4InhibitionDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "/tmp")
            + "/veith_cytochrome_p450_3a4_inhibition",
            download=True,
            transform=None,
            target_transform=None,
        )

    def time___init__(self, batch_size):
        VeithCytochromeP4503A4InhibitionDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", ".")
            + "/veith_cytochrome_p450_3a4_inhibition",
            download=True,
            transform=None,
            target_transform=None,
        )

    def peak_memory___init__(self, batch_size):
        VeithCytochromeP4503A4InhibitionDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", ".")
            + "/veith_cytochrome_p450_3a4_inhibition",
            download=True,
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
