from beignet.datasets import RandomEulerAngleDataset


class BenchRandomEulerAngleDataset:
    params = [
        [10, 100, 1000],
        ["XYZ"],
    ]

    param_names = ["batch_size", "axes"]

    def __init__(self):
        pass

    def setup(self, batch_size, axes):
        self.size = batch_size
        self.axes = axes
        self.dataset = RandomEulerAngleDataset(
            size=self.size,
            axes=self.axes,
            degrees=False,
            transform=None,
        )

    def time___init__(self, batch_size, axes):
        RandomEulerAngleDataset(
            size=self.size,
            axes=self.axes,
            degrees=False,
            transform=None,
        )

    def peak_memory___init__(self, batch_size, axes):
        RandomEulerAngleDataset(
            size=self.size,
            axes=self.axes,
            degrees=False,
            transform=None,
        )

    def time___len__(self, batch_size, axes):
        len(self.dataset)

    def peak_memory___len__(self, batch_size, axes):
        len(self.dataset)

    def time___getitem__(self, batch_size, axes):
        for i in range(min(batch_size, len(self.dataset))):
            self.dataset[i]

    def peak_memory___getitem__(self, batch_size, axes):
        for i in range(min(batch_size, len(self.dataset))):
            self.dataset[i]
