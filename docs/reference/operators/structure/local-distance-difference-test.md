# local_distance_difference_test

Compute the Local Distance Difference Test (LDDT) score for protein structure evaluation.

## Description

LDDT is a superposition-free score that evaluates how well the local distances between atoms are preserved in the predicted structure compared to the reference structure. It measures the fraction of atom pairs within a cutoff distance that have their distances preserved within specified thresholds.

The LDDT score is widely used in protein structure prediction assessment, including CASP (Critical Assessment of Structure Prediction) competitions and AlphaFold evaluations.

## Parameters

- **predicted_coords** (Tensor): Predicted atom coordinates with shape `(..., N, 3)`
- **reference_coords** (Tensor): Reference (true) atom coordinates with shape `(..., N, 3)`
- **atom_mask** (Tensor, optional): Binary mask indicating valid atoms with shape `(..., N)`. If None, all atoms are considered valid
- **cutoff** (float): Maximum distance cutoff in Angstroms for considering atom pairs. Default: 15.0
- **thresholds** (List[float]): Distance difference thresholds in Angstroms. The score is averaged over these thresholds. Default: [0.5, 1.0, 2.0, 4.0]
- **per_atom** (bool): If True, return per-atom LDDT scores. If False, return global average. Default: False

## Returns

- **lddt_score** (Tensor): LDDT scores in range [0, 1]. Shape is `(..., N)` if `per_atom=True`, otherwise `(...)`

## Examples

```python
import torch
import beignet

# Generate example coordinates
batch_size, n_atoms = 2, 100
predicted = torch.randn(batch_size, n_atoms, 3) * 10
reference = predicted + torch.randn_like(predicted) * 0.5
mask = torch.ones(batch_size, n_atoms)

# Calculate global LDDT score
lddt_global = beignet.local_distance_difference_test(
    predicted, reference, mask, per_atom=False
)
print(f"Global LDDT: {lddt_global}")  # Shape: (2,)

# Calculate per-atom LDDT scores
lddt_per_atom = beignet.local_distance_difference_test(
    predicted, reference, mask, per_atom=True
)
print(f"Per-atom LDDT shape: {lddt_per_atom.shape}")  # Shape: (2, 100)
```

## Notes

- LDDT is invariant to global rotations and translations
- Higher scores indicate better local structure preservation
- The default thresholds [0.5, 1.0, 2.0, 4.0] Å are standard in structure assessment
- LDDT uses hard thresholds, making it non-differentiable for gradient-based optimization

## References

- Mariani et al. (2013). lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests. Bioinformatics.