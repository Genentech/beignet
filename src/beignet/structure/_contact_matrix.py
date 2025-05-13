from typing import Callable

import torch
from torch import Tensor

from beignet import radius

from ._residue_array import ResidueArray


def _atom_thin_to_contact_matrix(
    atom_thin_xyz: Tensor,
    atom_thin_mask: Tensor,
    residue_mask_A: Tensor | None = None,
    residue_mask_B: Tensor | None = None,
    atom_mask_A: Tensor | None = None,
    atom_mask_B: Tensor | None = None,
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

    mask_A = atom_thin_mask
    if residue_mask_A is not None:
        mask_A = mask_A & residue_mask_A[..., None]
    if atom_mask_A is not None:
        mask_A = mask_A & atom_mask_A

    mask_B = atom_thin_mask
    if residue_mask_B is not None:
        mask_B = mask_B & residue_mask_B[..., None]
    if atom_mask_B is not None:
        mask_B = mask_B & atom_mask_B

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
    input: ResidueArray,
    residue_selector_A: Callable[[ResidueArray], Tensor] | Tensor | None = None,
    residue_selector_B: Callable[[ResidueArray], Tensor] | Tensor | None = None,
    atom_selector_A: Callable[[ResidueArray], Tensor] | Tensor | None = None,
    atom_selector_B: Callable[[ResidueArray], Tensor] | Tensor | None = None,
    radius_cutoff: float = 10.0,
    **residue_selector_kwargs,
) -> Tensor:
    if callable(residue_selector_A):
        residue_mask_A = residue_selector_A(input, **residue_selector_kwargs)
    elif isinstance(residue_selector_A, Tensor):
        residue_mask_A = residue_selector_A
    elif residue_selector_A is None:
        residue_mask_A = None
    else:
        raise AssertionError(f"{type(residue_selector_A)=} not supported")

    if callable(residue_selector_B):
        residue_mask_B = residue_selector_B(input, **residue_selector_kwargs)
    elif isinstance(residue_selector_B, Tensor):
        residue_mask_B = residue_selector_B
    elif residue_selector_B is None:
        residue_mask_B = None
    else:
        raise AssertionError(f"{type(residue_selector_B)=} not supported")

    if callable(atom_selector_A):
        atom_mask_A = atom_selector_A(input)
    elif isinstance(atom_selector_A, Tensor):
        atom_mask_A = atom_selector_A
    elif atom_selector_A is None:
        atom_mask_A = None
    else:
        raise AssertionError(f"{type(atom_selector_A)=} not supported")

    if callable(atom_selector_B):
        atom_mask_B = atom_selector_B(input)
    elif isinstance(atom_selector_B, Tensor):
        atom_mask_B = atom_selector_B
    elif atom_selector_B is None:
        atom_mask_B = None
    else:
        raise AssertionError(f"{type(atom_selector_B)=} not supported")

    return _atom_thin_to_contact_matrix(
        atom_thin_xyz=input.atom_thin_xyz,
        atom_thin_mask=input.atom_thin_mask,
        residue_mask_A=residue_mask_A,
        residue_mask_B=residue_mask_B,
        atom_mask_A=atom_mask_A,
        atom_mask_B=atom_mask_B,
        radius_cutoff=radius_cutoff,
    )
