import os

from beignet.datasets import TherapeuticAntibodyProfilerDataset


class BenchTherapeuticAntibodyProfilerDataset:
    params = [
        [10, 100, 1000],
    ]

    param_names = ["batch_size"]

    def __init__(self):
        pass

    def setup(self, batch_size):
        self.dataset = TherapeuticAntibodyProfilerDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", "/tmp")
            + "/therapeutic_antibody_profiler",
            download=True,
            transform=None,
            target_transform=None,
        )

    def time___init__(self, batch_size):
        TherapeuticAntibodyProfilerDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", ".")
            + "/therapeutic_antibody_profiler",
            download=True,
            transform=None,
            target_transform=None,
        )

    def peak_memory___init__(self, batch_size):
        TherapeuticAntibodyProfilerDataset(
            root=os.getenv("BEIGNET_BENCHMARKS_DATASET_ROOT", ".")
            + "/therapeutic_antibody_profiler",
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
