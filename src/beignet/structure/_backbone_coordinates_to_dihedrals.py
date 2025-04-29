import torch
from torch import Tensor

from beignet import dihedral_angle


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

    chain_boundary_mask = torch.diff(chain_id, n=1, dim=-1) == 0
    chain_break_mask = torch.diff(residue_index, n=1, dim=-1) == 1

    bb_n_xyz = backbone_coordinates[..., :, 0, :]
    bb_ca_xyz = backbone_coordinates[..., :, 1, :]
    bb_c_xyz = backbone_coordinates[..., :, 2, :]

    bb_n_mask = mask[..., :, 0]
    bb_ca_mask = mask[..., :, 1]
    bb_c_mask = mask[..., :, 2]

    phi = dihedral_angle(
        torch.stack(
            [
                bb_c_xyz[..., :-1, :],
                bb_n_xyz[..., 1:, :],
                bb_ca_xyz[..., 1:, :],
                bb_c_xyz[..., 1:, :],
            ],
            dim=-2,
        )
    )
    phi_mask = torch.stack(
        [
            bb_c_mask[..., :-1],
            bb_n_mask[..., 1:],
            bb_ca_mask[..., 1:],
            bb_c_mask[..., 1:],
        ],
        dim=-1,
    ).all(dim=-1)

    nan_tensor = torch.full_like(phi[..., :1], float("nan"))
    false_tensor = torch.zeros_like(bb_n_mask[..., :1])

    phi = torch.cat([nan_tensor, phi], dim=-1)
    phi_mask = (
        torch.cat([false_tensor, phi_mask], dim=-1)
        & torch.cat([false_tensor, chain_boundary_mask], dim=-1)
        & torch.cat([false_tensor, chain_break_mask], dim=-1)
    )

    psi = dihedral_angle(
        torch.stack(
            [
                bb_n_xyz[..., :-1, :],
                bb_ca_xyz[..., :-1, :],
                bb_c_xyz[..., :-1, :],
                bb_n_xyz[..., 1:, :],
            ],
            dim=-2,
        )
    )

    psi_mask = torch.stack(
        [
            bb_n_mask[..., :-1],
            bb_ca_mask[..., :-1],
            bb_c_mask[..., :-1],
            bb_n_mask[..., 1:],
        ],
        dim=-1,
    ).all(dim=-1)

    psi = torch.cat([psi, nan_tensor], dim=-1)
    psi_mask = (
        torch.cat([psi_mask, false_tensor], dim=-1)
        & torch.cat([chain_boundary_mask, false_tensor], dim=-1)
        & torch.cat([chain_break_mask, false_tensor], dim=-1)
    )

    omega = dihedral_angle(
        torch.stack(
            [
                bb_ca_xyz[..., :-1, :],
                bb_c_xyz[..., :-1, :],
                bb_n_xyz[..., 1:, :],
                bb_ca_xyz[..., 1:, :],
            ],
            dim=-2,
        )
    )

    omega_mask = torch.stack(
        [
            bb_ca_mask[..., :-1],
            bb_c_mask[..., :-1],
            bb_n_mask[..., 1:],
            bb_ca_mask[..., 1:],
        ],
        dim=-1,
    ).all(dim=-1)

    omega = torch.cat([omega, nan_tensor], dim=-1)
    omega_mask = (
        torch.cat([omega_mask, false_tensor], dim=-1)
        & torch.cat([chain_boundary_mask, false_tensor], dim=-1)
        & torch.cat([chain_break_mask, false_tensor], dim=-1)
    )

    dihedrals = torch.stack([phi, psi, omega], dim=-1)
    dihedrals_mask = torch.stack([phi_mask, psi_mask, omega_mask], dim=-1)
    dihedrals = torch.masked_fill(dihedrals, ~dihedrals_mask, torch.nan)

    return dihedrals, dihedrals_mask
