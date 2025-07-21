import torch

import beignet

from ._set_seed import set_seed


class BenchLocalDistanceDifferenceTest:
    params = [
        [50, 100, 200, 500],  # n_atoms
        [1, 4],  # batch_size
        [torch.float32, torch.float64],
        [True, False],  # per_atom
    ]

    param_names = ["n_atoms", "batch_size", "dtype", "per_atom"]

    def __init__(self):
        self.func = torch.compile(
            beignet.local_distance_difference_test,
            fullgraph=True,
        )

    def setup(self, n_atoms, batch_size, dtype, per_atom):
        set_seed()

        # Generate coordinates
        self.predicted_coords = torch.randn(batch_size, n_atoms, 3, dtype=dtype) * 10
        # Reference is slightly perturbed from predicted
        self.reference_coords = (
            self.predicted_coords + torch.randn_like(self.predicted_coords) * 0.5
        )

        # Atom mask (most atoms are valid)
        self.atom_mask = torch.rand(batch_size, n_atoms) > 0.1

        # Standard parameters
        self.cutoff = 15.0
        self.thresholds = [0.5, 1.0, 2.0, 4.0]
        self.per_atom = per_atom

    def time_local_distance_difference_test(self, n_atoms, batch_size, dtype, per_atom):
        self.func(
            self.predicted_coords,
            self.reference_coords,
            self.atom_mask,
            self.cutoff,
            self.thresholds,
            self.per_atom,
        )

    def peak_memory_local_distance_difference_test(
        self, n_atoms, batch_size, dtype, per_atom
    ):
        self.func(
            self.predicted_coords,
            self.reference_coords,
            self.atom_mask,
            self.cutoff,
            self.thresholds,
            self.per_atom,
        )
