import typing
from typing import Callable

import torch
from torch import Tensor

from beignet import radius

from ._invoke_selector import invoke_selector

if typing.TYPE_CHECKING:
    from ._residue_array import ResidueArray


def _atom_thin_to_contact_matrix(
    atom_thin_xyz: Tensor,
    atom_thin_mask: Tensor,
    mask_A: Tensor | None = None,
    mask_B: Tensor | None = None,
    radius_cutoff: float = 10.0,
):
    if atom_thin_xyz.ndim == 4:
        has_batch = True
    elif atom_thin_xyz.ndim == 3:
        has_batch = False
        # temporarily add batch dimension
        atom_thin_xyz = atom_thin_xyz[None]
        atom_thin_mask = atom_thin_mask[None]
    else:
        raise RuntimeError(f"{atom_thin_xyz.ndim=} not supported")

    B, L, _, _ = atom_thin_xyz.shape
    device = atom_thin_xyz.device

    if mask_A is None:
        mask_A = atom_thin_mask
    else:
        mask_A = mask_A & atom_thin_mask

    if mask_B is None:
        mask_B = atom_thin_mask
    else:
        mask_B = mask_B & atom_thin_mask

    batch_A, residue_A, _ = torch.nonzero(mask_A, as_tuple=True)
    batch_B, residue_B, _ = torch.nonzero(mask_B, as_tuple=True)

    atoms_A = atom_thin_xyz[mask_A]
    atoms_B = atom_thin_xyz[mask_B]

    row, col = radius(
        atoms_A,
        atoms_B,
        radius_cutoff,
        batch_x=batch_A,
        batch_y=batch_B,
        ignore_same_index=False,
    )

    #    r = torch.sqrt(torch.sum(torch.pow(atoms_A[col] - atoms_B[row], 2), dim=-1))
    #    assert r.max() <= radius_cutoff
    #    assert torch.equal(batch_A[col], batch_B[row])

    contact_matrix = torch.zeros(B, L, L, device=device, dtype=torch.int64)

    # NOTE this accumulates over atom indices
    contact_matrix = torch.index_put(
        contact_matrix,
        (batch_B[row], residue_B[row], residue_A[col]),
        torch.ones(
            row.shape[0], dtype=contact_matrix.dtype, device=contact_matrix.device
        ),
        accumulate=True,
    )

    # symmetrize
    contact_matrix = (contact_matrix + torch.transpose(contact_matrix, -2, -1)) > 0

    if not has_batch:
        contact_matrix = contact_matrix.view(L, L)

    return contact_matrix


def contact_matrix(
    input: "ResidueArray",
    selector_A: Callable[["ResidueArray"], Tensor] | Tensor | None = None,
    selector_B: Callable[["ResidueArray"], Tensor] | Tensor | None = None,
    radius_cutoff: float = 10.0,
    **selector_kwargs,
) -> Tensor:
    mask_A = invoke_selector(selector_A, input, **selector_kwargs)
    mask_B = invoke_selector(selector_B, input, **selector_kwargs)
    return _atom_thin_to_contact_matrix(
        atom_thin_xyz=input.atom_thin_xyz,
        atom_thin_mask=input.atom_thin_mask,
        mask_A=mask_A,
        mask_B=mask_B,
        radius_cutoff=radius_cutoff,
    )
