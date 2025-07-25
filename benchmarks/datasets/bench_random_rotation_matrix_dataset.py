from beignet.datasets import RandomRotationMatrixDataset

from .._set_seed import set_seed


class BenchRandomRotationMatrixDataset:
    params = [
        [10, 100, 1000],
    ]

    param_names = ["batch_size"]

    def setup(self, batch_size):
        set_seed()

        self.size = batch_size

        self.dataset = RandomRotationMatrixDataset(
            size=self.size,
            transform=None,
        )

    def time___init__(self, batch_size):
        RandomRotationMatrixDataset(
            size=self.size,
            transform=None,
        )

    def peak_memory___init__(self, batch_size):
        RandomRotationMatrixDataset(
            size=self.size,
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
