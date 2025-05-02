from functools import cache

import einops
import torch
from torch import Tensor

from beignet.constants import AMINO_ACID_1_TO_3, ATOM_THIN_ATOMS, STANDARD_RESIDUES

# 180 degree symmetry
_SYMMETRIC_ATOM_SWAPS = {
    "ASP": {"OD1": "OD2"},
    "GLU": {"OE1": "OE2"},
    "PHE": {"CD1": "CD2", "CE1": "CE2"},
    "TYR": {"CD1": "CD2", "CE1": "CE2"},
}


@cache
def _make_symmetric_atom_swap_indices() -> list[list[int]]:
    out = []
    for res in STANDARD_RESIDUES:
        atoms = ATOM_THIN_ATOMS[AMINO_ACID_1_TO_3[res]]
        res_swaps = _SYMMETRIC_ATOM_SWAPS.get(AMINO_ACID_1_TO_3[res], {})
        res_swaps = res_swaps | dict((v, k) for k, v in res_swaps.items())
        indices = [
            atoms.index(res_swaps.get(a, a)) if a else i for i, a in enumerate(atoms)
        ]
        out.append(indices)
    return out


@cache
def _make_atom_thin_is_symmetric_mask() -> list[int]:
    out = []
    for res in STANDARD_RESIDUES:
        atoms = ATOM_THIN_ATOMS[AMINO_ACID_1_TO_3[res]]
        res_swaps = _SYMMETRIC_ATOM_SWAPS.get(AMINO_ACID_1_TO_3[res], {})
        res_swaps = res_swaps | dict((v, k) for k, v in res_swaps.items())
        mask = [atom in res_swaps for atom in atoms]
        out.append(mask)
    return out


def swap_symmetric_atom_thin_atoms(
    residue_type: Tensor, atom_thin_xyz: Tensor, atom_thin_mask: Tensor | None = None
) -> tuple[Tensor, Tensor | None]:
    symmetric_swap_indices = torch.as_tensor(
        _make_symmetric_atom_swap_indices(), device=residue_type.device
    )

    symmetric_swap_indices = symmetric_swap_indices[residue_type]

    atom_thin_xyz = torch.gather(
        atom_thin_xyz,
        dim=-2,
        index=einops.repeat(symmetric_swap_indices, "... -> ... 3"),
    )

    if atom_thin_mask is not None:
        atom_thin_mask = torch.gather(
            atom_thin_mask,
            dim=-1,
            index=symmetric_swap_indices,
        )

    return atom_thin_xyz, atom_thin_mask


def rename_symmetric_atoms(
    residue_type: Tensor,
    atom_thin_xyz: Tensor,
    atom_thin_mask: Tensor,
    atom_thin_xyz_reference: Tensor,
    eps: float = 1e-12,
):
    atom_thin_xyz_alt, _ = swap_symmetric_atom_thin_atoms(
        residue_type, atom_thin_xyz, atom_thin_mask
    )

    dist = torch.sqrt(
        torch.sum(
            torch.pow(
                atom_thin_xyz[..., None, :, None, :]
                - atom_thin_xyz[..., None, :, None, :, :],
                2,
            ),
            dim=-1,
        )
        + 1e-12
    )

    dist_alt = torch.sqrt(
        torch.sum(
            torch.pow(
                atom_thin_xyz_alt[..., None, :, None, :]
                - atom_thin_xyz_alt[..., None, :, None, :, :],
                2,
            ),
            dim=-1,
        )
        + 1e-12
    )

    dist_reference = torch.sqrt(
        torch.sum(
            torch.pow(
                atom_thin_xyz_reference[..., None, :, None, :]
                - atom_thin_xyz_reference[..., None, :, None, :, :],
                2,
            ),
            dim=-1,
        )
        + 1e-12
    )

    lddt = torch.sqrt(eps + torch.pow((dist - dist_reference), 2))
    lddt_alt = torch.sqrt(eps + torch.pow((dist_alt - dist_reference), 2))

    atom_thin_is_symmetric = torch.as_tensor(_make_atom_thin_is_symmetric_mask())[
        residue_type
    ]
    pair_mask = (
        atom_thin_is_symmetric[..., None, :, None] & atom_thin_mask[..., None, :, None]
    ) & (
        ~atom_thin_is_symmetric[..., None, :, None, :]
        & atom_thin_mask[..., None, :, None, :]
    )

    per_res_lddt = torch.sum(pair_mask * lddt, dim=(-1, -2, -3))
    per_res_lddt_alt = torch.sum(pair_mask * lddt_alt, dim=(-1, -2, -3))

    return torch.where(
        (per_res_lddt < per_res_lddt_alt)[..., None, None],
        atom_thin_xyz,
        atom_thin_xyz_alt,
    )
