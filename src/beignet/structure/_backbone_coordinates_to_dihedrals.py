import torch
from torch import Tensor

from beignet import dihedral


def backbone_coordinates_to_dihedrals(
    backbone_coordinates: Tensor,
    mask: Tensor,
    residue_index: Tensor | None = None,
    chain_id: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    assert backbone_coordinates.ndim >= 3
    L = backbone_coordinates.shape[-3]

    if residue_index is None:
        residue_index = torch.arange(L).expand(
            *backbone_coordinates.shape[:-3], -1
        )  # [..., L]

    if chain_id is None:
        chain_id = torch.zeros_like(
            backbone_coordinates[..., :, 0, 0], dtype=torch.int64
        )  # [..., L]

    chain_boundary_mask = torch.cat(
        [
            torch.zeros_like(chain_id[..., :1], dtype=torch.bool),
            torch.diff(chain_id, n=1, dim=-1) == 0,
        ],
        dim=-1,
    )

    chain_break_mask = torch.cat(
        [
            torch.zeros_like(residue_index[..., :1], dtype=torch.bool),
            torch.diff(residue_index, n=1, dim=-1) == 1,
        ],
        dim=-1,
    )

    N = backbone_coordinates[..., :, 0, :]
    CA = backbone_coordinates[..., :, 1, :]
    C = backbone_coordinates[..., :, 2, :]

    N_mask = mask[..., :, 0]
    CA_mask = mask[..., :, 1]
    C_mask = mask[..., :, 2]

    phi = dihedral(
        torch.stack(
            [C[..., :-1, :], N[..., 1:, :], CA[..., 1:, :], C[..., 1:, :]], dim=-2
        )
    )
    phi_mask = torch.stack(
        [C_mask[..., :-1], N_mask[..., 1:], CA_mask[..., 1:], C_mask[..., 1:]], dim=-1
    ).all(dim=-1)

    nan_tensor = torch.full_like(phi[..., :1], float("nan"))
    false_tensor = torch.zeros_like(N_mask[..., :1])

    phi = torch.cat([nan_tensor, phi], dim=-1)
    phi_mask = (
        torch.cat([false_tensor, phi_mask], dim=-1)
        & chain_boundary_mask
        & chain_break_mask
    )

    psi = dihedral(
        torch.stack(
            [N[..., :-1, :], CA[..., :-1, :], C[..., :-1, :], N[..., 1:, :]], dim=-2
        )
    )

    psi_mask = torch.stack(
        [N_mask[..., :-1], CA_mask[..., :-1], C_mask[..., :-1], N_mask[..., 1:]], dim=-1
    ).all(dim=-1)

    psi = torch.cat([psi, nan_tensor], dim=-1)
    psi_mask = (
        torch.cat([psi_mask, false_tensor], dim=-1)
        & chain_boundary_mask
        & chain_break_mask
    )

    omega = dihedral(
        torch.stack(
            [CA[..., :-1, :], C[..., :-1, :], N[..., 1:, :], CA[..., 1:, :]], dim=-2
        )
    )

    omega_mask = torch.stack(
        [CA_mask[..., :-1], C_mask[..., :-1], N_mask[..., 1:], CA_mask[..., 1:]], dim=-1
    ).all(dim=-1)

    omega = torch.cat([omega, nan_tensor], dim=-1)
    omega_mask = (
        torch.cat([omega_mask, false_tensor], dim=-1)
        & chain_boundary_mask
        & chain_break_mask
    )

    dihedrals = torch.stack([phi, psi, omega], dim=-1)
    dihedrals_mask = torch.stack([phi_mask, psi_mask, omega_mask], dim=-1)

    return dihedrals, dihedrals_mask
