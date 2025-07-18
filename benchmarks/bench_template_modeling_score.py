import os

import torch

import beignet

# Ensure reproducibility
SEED = int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42))
torch.manual_seed(SEED)


class TimeTemplateModeingScore:
    """Benchmark TM-score calculation time."""

    params = ([10, 50, 100, 500], [torch.float32, torch.float64], [False, True])
    param_names = ["n_residues", "dtype", "aligned"]

    def setup(self, n_residues, dtype, aligned):
        self.structure1 = torch.randn(n_residues, 3, dtype=dtype)
        self.structure2 = self.structure1 + 0.5 * torch.randn(
            n_residues, 3, dtype=dtype
        )

        # Compile the function
        self.compiled_fn = torch.compile(
            beignet.template_modeling_score, fullgraph=True
        )

        # Warmup
        for _ in range(3):
            self.compiled_fn(self.structure1, self.structure2, aligned=aligned)

    def time_template_modeling_score(self, n_residues, dtype, aligned):
        self.compiled_fn(self.structure1, self.structure2, aligned=aligned)


class TimeTemplateModeingScoreBatched:
    """Benchmark batched TM-score calculation time."""

    params = ([8, 32], [50, 100, 200], [torch.float32, torch.float64])
    param_names = ["batch_size", "n_residues", "dtype"]

    def setup(self, batch_size, n_residues, dtype):
        self.structure1 = torch.randn(batch_size, n_residues, 3, dtype=dtype)
        self.structure2 = self.structure1 + 0.5 * torch.randn(
            batch_size, n_residues, 3, dtype=dtype
        )

        # Compile the function
        self.compiled_fn = torch.compile(
            beignet.template_modeling_score, fullgraph=True
        )

        # Warmup
        for _ in range(3):
            self.compiled_fn(self.structure1, self.structure2)

    def time_template_modeling_score_batched(self, batch_size, n_residues, dtype):
        self.compiled_fn(self.structure1, self.structure2)


class TimeTemplateModeingScoreWithWeights:
    """Benchmark TM-score calculation with weights."""

    params = ([50, 100, 200], [torch.float32, torch.float64])
    param_names = ["n_residues", "dtype"]

    def setup(self, n_residues, dtype):
        self.structure1 = torch.randn(n_residues, 3, dtype=dtype)
        self.structure2 = self.structure1 + 0.5 * torch.randn(
            n_residues, 3, dtype=dtype
        )
        self.weights = torch.rand(n_residues, dtype=dtype)

        # Compile the function
        self.compiled_fn = torch.compile(
            beignet.template_modeling_score, fullgraph=True
        )

        # Warmup
        for _ in range(3):
            self.compiled_fn(self.structure1, self.structure2, weights=self.weights)

    def time_template_modeling_score_weighted(self, n_residues, dtype):
        self.compiled_fn(self.structure1, self.structure2, weights=self.weights)


class PeakMemoryTemplateModeingScore:
    """Benchmark peak memory usage for TM-score calculation."""

    params = ([100, 500, 1000], [torch.float32, torch.float64])
    param_names = ["n_residues", "dtype"]

    def setup(self, n_residues, dtype):
        self.structure1 = torch.randn(n_residues, 3, dtype=dtype)
        self.structure2 = self.structure1 + 0.5 * torch.randn(
            n_residues, 3, dtype=dtype
        )

        # Compile the function
        self.compiled_fn = torch.compile(
            beignet.template_modeling_score, fullgraph=True
        )

    def peakmem_template_modeling_score(self, n_residues, dtype):
        self.compiled_fn(self.structure1, self.structure2)


class PeakMemoryTemplateModeingScoreBatched:
    """Benchmark peak memory usage for batched TM-score calculation."""

    params = ([16, 32], [100, 200], [torch.float32, torch.float64])
    param_names = ["batch_size", "n_residues", "dtype"]

    def setup(self, batch_size, n_residues, dtype):
        self.structure1 = torch.randn(batch_size, n_residues, 3, dtype=dtype)
        self.structure2 = self.structure1 + 0.5 * torch.randn(
            batch_size, n_residues, 3, dtype=dtype
        )

        # Compile the function
        self.compiled_fn = torch.compile(
            beignet.template_modeling_score, fullgraph=True
        )

    def peakmem_template_modeling_score_batched(self, batch_size, n_residues, dtype):
        self.compiled_fn(self.structure1, self.structure2)
