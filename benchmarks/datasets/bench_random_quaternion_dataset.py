from beignet.datasets import RandomQuaternionDataset


class BenchRandomQuaternionDataset:
    params = [
        [10, 100, 1000, 10000],
        [True, False],
    ]

    param_names = ["batch_size", "canonical"]

    def __init__(self):
        pass

    def setup(self, batch_size, canonical):
        self.size = batch_size

        self.canonical = canonical

        self.dataset = RandomQuaternionDataset(
            size=self.size,
            canonical=self.canonical,
            transform=None,
        )

    def time___init__(self, batch_size, canonical):
        RandomQuaternionDataset(
            size=self.size,
            canonical=self.canonical,
            transform=None,
        )

    def peak_memory___init__(self, batch_size, canonical):
        RandomQuaternionDataset(
            size=self.size,
            canonical=self.canonical,
            transform=None,
        )

    def time___len__(self, batch_size, canonical):
        len(self.dataset)

    def peak_memory___len__(self, batch_size, canonical):
        len(self.dataset)

    def time___getitem__(self, batch_size, canonical):
        for i in range(min(batch_size, len(self.dataset))):
            self.dataset[i]

    def peak_memory___getitem__(self, batch_size, canonical):
        for i in range(min(batch_size, len(self.dataset))):
            self.dataset[i]
