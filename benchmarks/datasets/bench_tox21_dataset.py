from beignet.datasets import Tox21Dataset


class Tox21DatasetBenchmark:
    params = [
        [10, 100, 1000],
    ]

    param_names = ["batch_size"]

    def __init__(self):
        pass

    def setup(self, batch_size):
        self.dataset = Tox21Dataset(
            root="/tmp/tox21",
            download=True,
            transform=None,
            target_transform=None,
        )

    def time___init__(self, batch_size):
        Tox21Dataset(
            root="/tmp/tox21",
            download=True,
            transform=None,
            target_transform=None,
        )

    def time___len__(self, batch_size):
        len(self.dataset)

    def time___getitem__(self, batch_size):
        for i in range(min(batch_size, len(self.dataset))):
            self.dataset[i]

    def peak_memory___init__(self, batch_size):
        Tox21Dataset(
            root="/tmp/tox21",
            download=True,
            transform=None,
            target_transform=None,
        )

    def peak_memory___getitem__(self, batch_size):
        for i in range(min(batch_size, len(self.dataset))):
            self.dataset[i]
