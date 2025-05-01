from functools import cache

import einops
import torch
from torch import Tensor

from beignet.constants import AMINO_ACID_1_TO_3, ATOM_THIN_ATOMS, STANDARD_RESIDUES

# 180 degree symmetry
_AMBIGUOUS_ATOM_SWAPS = {
    "ASP": {"OD1": "OD2"},
    "GLU": {"OE1": "OE2"},
    "PHE": {"CD1": "CD2", "CE1": "CE2"},
    "TYR": {"CD1": "CD2", "CE1": "CE2"},
}


@cache
def _make_ambiguous_atom_swap_indices() -> list[list]:
    out = []
    for res in STANDARD_RESIDUES:
        atoms = ATOM_THIN_ATOMS[AMINO_ACID_1_TO_3[res]]
        res_swaps = _AMBIGUOUS_ATOM_SWAPS.get(AMINO_ACID_1_TO_3[res], {})
        res_swaps = res_swaps | dict((v, k) for k, v in res_swaps.items())
        indices = [
            atoms.index(res_swaps.get(a, a)) if a else i for i, a in enumerate(atoms)
        ]
        out.append(indices)
    return out


def swap_ambiguous_atom_thin_atoms(
    atom_thin_xyz: Tensor, residue_type: Tensor, atom_thin_mask: Tensor | None = None
) -> tuple[Tensor, Tensor | None]:
    ambiguous_swap_indices = torch.as_tensor(
        _make_ambiguous_atom_swap_indices(), device=residue_type.device
    )

    ambiguous_swap_indices = ambiguous_swap_indices[residue_type]

    atom_thin_xyz = torch.gather(
        atom_thin_xyz,
        dim=-2,
        index=einops.repeat(ambiguous_swap_indices, "... -> ... 3"),
    )

    if atom_thin_mask is not None:
        atom_thin_mask = torch.gather(
            atom_thin_mask,
            dim=-1,
            index=ambiguous_swap_indices,
        )

    return atom_thin_xyz, atom_thin_mask
