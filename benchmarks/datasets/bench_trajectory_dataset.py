import os

from beignet.datasets import TrajectoryDataset

from .._set_seed import set_seed


class BenchTrajectoryDataset:
    params = [
        [10, 100, 1000],
    ]

    param_names = ["batch_size"]

    def setup(self, batch_size):
        set_seed()

        self.dataset = TrajectoryDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "/tmp")
            + "/trajectory_dataset",
            transform=None,
        )

    def time___init__(self, batch_size):
        TrajectoryDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", ".")
            + "/trajectory_dataset",
            transform=None,
        )

    def peak_memory___init__(self, batch_size):
        TrajectoryDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", ".")
            + "/trajectory_dataset",
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
