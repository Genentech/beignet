import dataclasses
from typing import Callable, Literal

import einops
import torch
from biotite.structure.io import pdbx as pdbx
from torch import Tensor

from beignet.constants import ATOM_THIN_ATOMS, STANDARD_RESIDUES

from ._rename_symmetric_atoms import rename_symmetric_atoms as _rename_symmetric_atoms
from ._residue_array import ResidueArray
from ._rigid import Rigid

restypes_with_x = STANDARD_RESIDUES + ["X"]
restype_order_with_x = {r: i for i, r in enumerate(restypes_with_x)}
n_atom_thin = len(ATOM_THIN_ATOMS["ALA"])


def _atom_name_mask(atom_name: str, device=None) -> Tensor:
    return torch.stack(
        [
            (
                torch.nn.functional.one_hot(
                    torch.as_tensor(v.index(atom_name), device=device), n_atom_thin
                )
                if atom_name in v
                else torch.zeros(n_atom_thin, device=device, dtype=torch.int64)
            )
            for v in ATOM_THIN_ATOMS.values()
        ]
    )


def superimpose_atom_thin(
    fixed_atom_thin_xyz: Tensor,
    mobile_atom_thin_xyz: Tensor,
    atom_thin_mask: Tensor,
) -> tuple[Tensor, Rigid]:
    x = einops.rearrange(mobile_atom_thin_xyz, "... l n d -> ... (l n) d")
    y = einops.rearrange(fixed_atom_thin_xyz, "... l n d -> ... (l n) d")
    mask = einops.rearrange(atom_thin_mask, "... l n -> ... (l n)")

    T = Rigid.kabsch(y, x, weights=mask, keepdim=True)
    x = T(x)

    aligned_atom_thin_xyz = einops.rearrange(
        x, "... (l n) d -> ... l n d", n=n_atom_thin
    )

    return aligned_atom_thin_xyz, T


def rmsd_atom_thin(
    atom_thin_xyz: Tensor,
    target_atom_thin_xyz: Tensor,
    atom_thin_mask: Tensor,
) -> Tensor:
    delta2 = torch.square(atom_thin_xyz - target_atom_thin_xyz)
    a = einops.reduce(delta2 * atom_thin_mask[..., None], "... l n d -> ...", "sum")
    b = einops.reduce(atom_thin_mask, "... l n -> ...", "sum")
    rmsd = torch.sqrt(a / b)
    return rmsd


def rmsd(
    input: ResidueArray,
    target: ResidueArray,
    residue_selector: Callable[[ResidueArray], Tensor] | None = None,
    atom_selector: Literal["c_alpha", "all"] = "all",
    rename_symmetric_atoms: bool = True,
    **residue_selector_kwargs,
) -> Tensor:
    match atom_selector:
        case "c_alpha":
            atom_thin_mask = _atom_name_mask("CA", device=input.residue_type.device)[
                input.residue_type
            ]
        case "all":
            atom_thin_mask = torch.ones_like(input.atom_thin_mask)
        case _:
            raise AssertionError(f"{atom_selector=} not supported")

    if residue_selector is not None:
        residue_mask = residue_selector(input, **residue_selector_kwargs)
    else:
        residue_mask = torch.ones_like(input.padding_mask)

    atom_thin_mask = atom_thin_mask & input.atom_thin_mask
    atom_thin_mask = atom_thin_mask & target.atom_thin_mask
    atom_thin_mask = atom_thin_mask & residue_mask[..., None]

    return rmsd_atom_thin(
        atom_thin_xyz=input.atom_thin_xyz,
        target_atom_thin_xyz=target.atom_thin_xyz,
        atom_thin_mask=atom_thin_mask,
    )


def superimpose(
    fixed: ResidueArray,
    mobile: ResidueArray,
    residue_selector: Callable[[ResidueArray], Tensor] | None = None,
    atom_selector: Literal["c_alpha", "all"] = "all",
    rename_symmetric_atoms: bool = True,
    **residue_selector_kwargs,
) -> tuple[ResidueArray, Rigid, Tensor]:
    match atom_selector:
        case "c_alpha":
            atom_thin_mask = _atom_name_mask("CA", device=mobile.residue_type.device)[
                mobile.residue_type
            ]
        case "all":
            atom_thin_mask = torch.ones_like(mobile.atom_thin_mask)
        case _:
            raise AssertionError(f"{atom_selector=} not supported")

    if residue_selector is not None:
        residue_mask = residue_selector(mobile, **residue_selector_kwargs)
    else:
        residue_mask = torch.ones_like(mobile.padding_mask)

    if rename_symmetric_atoms:
        mobile_atom_thin_xyz = _rename_symmetric_atoms(
            mobile.residue_type,
            mobile.atom_thin_xyz,
            mobile.atom_thin_mask,
            fixed.atom_thin_xyz,
        )
    else:
        mobile_atom_thin_xyz = mobile.atom_thin_xyz

    atom_thin_mask = atom_thin_mask & mobile.atom_thin_mask
    atom_thin_mask = atom_thin_mask & fixed.atom_thin_mask
    atom_thin_mask = atom_thin_mask & residue_mask[..., None]

    aligned_atom_thin_xyz, T = superimpose_atom_thin(
        fixed_atom_thin_xyz=fixed.atom_thin_xyz,
        mobile_atom_thin_xyz=mobile_atom_thin_xyz,
        atom_thin_mask=atom_thin_mask,
    )

    rmsd = rmsd_atom_thin(
        atom_thin_xyz=aligned_atom_thin_xyz,
        target_atom_thin_xyz=fixed.atom_thin_xyz,
        atom_thin_mask=atom_thin_mask,
    )

    aligned = dataclasses.replace(mobile, atom_thin_xyz=aligned_atom_thin_xyz)

    return aligned, T, rmsd
