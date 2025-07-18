import os

import torch

import beignet

torch.manual_seed(int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42)))


class TimeGlobalDistanceTestTotalScore:
    params = ([1, 10, 100], [100, 500, 1000], [torch.float32, torch.float64])
    param_names = ["batch_size", "n_residues", "dtype"]

    def setup(self, batch_size, n_residues, dtype):
        self.input = torch.randn(batch_size, n_residues, 3, dtype=dtype)
        self.reference = torch.randn(batch_size, n_residues, 3, dtype=dtype)
        self.mask = torch.rand(batch_size, n_residues) > 0.1
        self.fn = torch.compile(
            beignet.global_distance_test_total_score, fullgraph=True
        )

    def time_global_distance_test_total_score(self, batch_size, n_residues, dtype):
        self.fn(self.input, self.reference)

    def time_global_distance_test_total_score_with_mask(
        self, batch_size, n_residues, dtype
    ):
        self.fn(self.input, self.reference, mask=self.mask)


class PeakMemGlobalDistanceTestTotalScore:
    params = ([1, 10, 100], [100, 500, 1000], [torch.float32, torch.float64])
    param_names = ["batch_size", "n_residues", "dtype"]

    def setup(self, batch_size, n_residues, dtype):
        self.input = torch.randn(batch_size, n_residues, 3, dtype=dtype)
        self.reference = torch.randn(batch_size, n_residues, 3, dtype=dtype)
        self.mask = torch.rand(batch_size, n_residues) > 0.1
        self.fn = torch.compile(
            beignet.global_distance_test_total_score, fullgraph=True
        )

    def peakmem_global_distance_test_total_score(self, batch_size, n_residues, dtype):
        self.fn(self.input, self.reference)

    def peakmem_global_distance_test_total_score_with_mask(
        self, batch_size, n_residues, dtype
    ):
        self.fn(self.input, self.reference, mask=self.mask)
