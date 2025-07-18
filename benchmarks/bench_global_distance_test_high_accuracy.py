import os

import torch

import beignet

torch.manual_seed(int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42)))


class TimeGlobalDistanceTestHighAccuracy:
    params = ([1, 10, 100], [100, 500, 1000], [torch.float32, torch.float64])
    param_names = ["batch_size", "n_residues", "dtype"]

    def setup(self, batch_size, n_residues, dtype):
        self.input = torch.randn(batch_size, n_residues, 3, dtype=dtype)
        self.reference = torch.randn(batch_size, n_residues, 3, dtype=dtype)
        self.mask = torch.rand(batch_size, n_residues) > 0.1
        self.fn = torch.compile(
            beignet.global_distance_test_high_accuracy, fullgraph=True
        )

    def time_global_distance_test_high_accuracy(self, batch_size, n_residues, dtype):
        self.fn(self.input, self.reference)

    def time_global_distance_test_high_accuracy_with_mask(
        self, batch_size, n_residues, dtype
    ):
        self.fn(self.input, self.reference, mask=self.mask)


class PeakMemGlobalDistanceTestHighAccuracy:
    params = ([1, 10, 100], [100, 500, 1000], [torch.float32, torch.float64])
    param_names = ["batch_size", "n_residues", "dtype"]

    def setup(self, batch_size, n_residues, dtype):
        self.input = torch.randn(batch_size, n_residues, 3, dtype=dtype)
        self.reference = torch.randn(batch_size, n_residues, 3, dtype=dtype)
        self.mask = torch.rand(batch_size, n_residues) > 0.1
        self.fn = torch.compile(
            beignet.global_distance_test_high_accuracy, fullgraph=True
        )

    def peakmem_global_distance_test_high_accuracy(self, batch_size, n_residues, dtype):
        self.fn(self.input, self.reference)

    def peakmem_global_distance_test_high_accuracy_with_mask(
        self, batch_size, n_residues, dtype
    ):
        self.fn(self.input, self.reference, mask=self.mask)
