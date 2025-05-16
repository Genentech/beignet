import dataclasses
import typing
from typing import Callable

import einops
import torch
from biotite.structure.io import pdbx as pdbx
from torch import Tensor

from beignet.constants import ATOM_THIN_ATOMS, STANDARD_RESIDUES

from ._invoke_selector import invoke_selector
from ._rename_symmetric_atoms import rename_symmetric_atoms as _rename_symmetric_atoms
from ._rigid import Rigid
from .selectors import AllSelector

if typing.TYPE_CHECKING:
    from ._residue_array import ResidueArray

restypes_with_x = STANDARD_RESIDUES + ["X"]
restype_order_with_x = {r: i for i, r in enumerate(restypes_with_x)}
n_atom_thin = len(ATOM_THIN_ATOMS["ALA"])


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
    input: "ResidueArray",
    target: "ResidueArray",
    selector: Callable[["ResidueArray"], Tensor] | Tensor | None = None,
    **selector_kwargs,
) -> Tensor:
    """
    Calculate the RMSD between two residue arrays.

    Parameters
    ----------
    input : ResidueArray
        Input residue array for comparison.
    target : ResidueArray
        Target residue array to compare against.
    selector : Callable[[ResidueArray], Tensor] | Tensor | None, optional
        Selector function or mask to determine which atoms to include in the RMSD calculation.
        If None, all atoms are used (default: None).
    **selector_kwargs
        Additional keyword arguments to pass to the selector function.

    Returns
    -------
    Tensor
        The calculated RMSD value between the input and target structures.

    Notes
    -----
    RMSD is calculated only on atoms that are present in both structures and match
    the selection criteria.
    """
    if selector is None:
        selector = AllSelector()
    atom_thin_mask = invoke_selector(selector, input, **selector_kwargs)

    atom_thin_mask = atom_thin_mask & input.atom_thin_mask
    atom_thin_mask = atom_thin_mask & target.atom_thin_mask

    return rmsd_atom_thin(
        atom_thin_xyz=input.atom_thin_xyz,
        target_atom_thin_xyz=target.atom_thin_xyz,
        atom_thin_mask=atom_thin_mask,
    )


def superimpose(
    fixed: "ResidueArray",
    mobile: "ResidueArray",
    selector: Callable[["ResidueArray"], Tensor] | Tensor | None = None,
    rename_symmetric_atoms: bool = True,
    **selector_kwargs,
) -> tuple["ResidueArray", Rigid, Tensor]:
    """
    Superimpose a mobile structure onto a fixed reference structure.

    Parameters
    ----------
    fixed : ResidueArray
        The reference structure that remains fixed.
    mobile : ResidueArray
        The structure to be superimposed onto the fixed structure.
    selector : Callable[[ResidueArray], Tensor] | Tensor | None, optional
        Selector function or mask to determine which atoms to use for superposition.
        If None, all atoms are used (default: None).
    rename_symmetric_atoms : bool, optional
        Whether to rename symmetric atoms to match between structures (default: True).
        This improves superposition for residues with symmetric side chains.
    **selector_kwargs
        Additional keyword arguments to pass to the selector function.

    Returns
    -------
    tuple[ResidueArray, Rigid, Tensor]
        A tuple containing:
        - aligned : ResidueArray
            The mobile structure after superposition onto the fixed structure.
        - T : Rigid
            The rigid transformation applied to the mobile structure.
        - rmsd : Tensor
            The RMSD between the aligned structure and the fixed structure.
    """
    if selector is None:
        selector = AllSelector()
    atom_thin_mask = invoke_selector(selector, fixed, **selector_kwargs)

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
