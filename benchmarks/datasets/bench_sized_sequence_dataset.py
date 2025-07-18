import os

import numpy as np

from beignet.datasets import SizedSequenceDataset

from .._set_seed import set_seed


class BenchSizedSequenceDataset:
    params = [
        [10, 100, 1000],
    ]

    param_names = ["batch_size"]

    def setup(self, batch_size):
        set_seed()

        self.dataset = SizedSequenceDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "/tmp")
            + "/sized_sequence_dataset",
            sizes=np.random.randint(10, 1000, size=batch_size),
        )

    def time___init__(self, batch_size):
        SizedSequenceDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", ".")
            + "/sized_sequence_dataset",
            sizes=np.random.randint(10, 1000, size=batch_size),
        )

    def peak_memory___init__(self, batch_size):
        SizedSequenceDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", ".")
            + "/sized_sequence_dataset",
            sizes=np.random.randint(10, 1000, size=batch_size),
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
