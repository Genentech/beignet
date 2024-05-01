from typing import Sequence

import torch
from torch import Tensor

from beignet.constants import (
    ADJACENT_RESIDUE_PHI_COSINE,
    ADJACENT_RESIDUE_PSI_COSINE,
    AMINO_ACID_3,
)


def bond_length_violation_loss(
    pred_atom_positions: Tensor,  # (*, N, 37/14, 3)
    pred_atom_mask: Tensor,  # (*, N, 37/14)
    residue_index: Tensor,  # (*, N)
    amino_acid: Tensor,  # (*, N)
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
) -> dict[str, Tensor]:
    r"""
    Parameters
    ----------
    pred_atom_positions : Tensor, shape=(*, N, 37/14, 3)
        Atom positions in atom37/14 representation.
    pred_atom_mask : Tensor, shape=(*, N, 37/14)
        Atom mask in atom37/14 representation.
    residue_index : Tensor, shape=(*, N)
        Residue index for given amino acid, this is assumed to be monotonically
        increasing.
    amino_acid : Tensor, shape=(*, N)
        Amino acid type of given residue.
    tolerance_factor_soft : float, optional
        Soft tolerance factor measured in standard deviations of pdb
        distributions. Default, 12.0.
    tolerance_factor_hard : float, optional
        Hard tolerance factor measured in standard deviations of pdb
        distributions. Default, 12.0.
    eps : float, optional
        Small value to avoid division by zero. Default, 1e-6.

    Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

    Returns:
      Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
            by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
            by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation
            present.
    """
    error_target_0 = ADJACENT_RESIDUE_PSI_COSINE[0]

    error_target_1 = [0.014, 0.016][0]
    error_target_3 = ADJACENT_RESIDUE_PHI_COSINE[0]
    error_target_4 = ADJACENT_RESIDUE_PHI_COSINE[1]

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = (
        amino_acid[..., 1:]
        == {k: v for v, k in enumerate([*AMINO_ACID_3, "UNK"])}["PRO"]
    )

    gt_length = _gt_length(next_is_proline)
    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    this_c_pos = pred_atom_positions[..., :-1, 2, :]
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :]
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]

    has_no_gap_mask = residue_index[..., 1:] - residue_index[..., :-1] == 1.0

    bond_length_0 = _bond_length(next_n_pos, this_c_pos)

    unit_vector_0 = (next_n_pos - this_c_pos) / bond_length_0[..., None]

    error_0 = _error(bond_length_0, gt_length)
    loss_per_residue_0 = _loss_per_residue(
        error_0, _gt_stddev(next_is_proline), tolerance_factor_soft
    )
    loss_0 = _loss(loss_per_residue_0, this_c_mask * next_n_mask * has_no_gap_mask)

    bond_length_1 = _bond_length(this_c_pos, this_ca_pos)
    bond_length_2 = _bond_length(next_ca_pos, next_n_pos)

    ca_c_n_cos_angle = torch.sum(
        (this_ca_pos - this_c_pos) / bond_length_1[..., None] * unit_vector_0, dim=-1
    )

    error_1 = _error(ca_c_n_cos_angle, error_target_0)
    loss_per_residue_1 = _loss_per_residue(
        error_1, error_target_1, tolerance_factor_soft
    )
    loss_1 = _loss(
        loss_per_residue_1, this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    )

    c_n_ca_cos_angle = _c_n_ca_cos_angle(
        unit_vector_0, (next_ca_pos - next_n_pos) / bond_length_2[..., None]
    )
    error_2 = _error(c_n_ca_cos_angle, error_target_3)

    loss_per_residue_2 = _loss_per_residue(
        error_2, error_target_4, tolerance_factor_soft
    )
    loss_2 = _loss(
        loss_per_residue_2, this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    )

    per_residue_loss_sum = _per_residue_loss_sum(
        loss_per_residue_2, loss_per_residue_0, loss_per_residue_1
    )

    violation_mask = _per_residue_violation_mask(
        [error_0, error_2, error_1],
        error_target_4,
        this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask,
        tolerance_factor_hard,
    )

    return {
        "c_n_loss_mean": loss_0,
        "ca_c_n_loss_mean": loss_1,
        "c_n_ca_loss_mean": loss_2,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
    }


def _c_n_ca_cos_angle(input, other):
    return torch.sum(-input * other, dim=-1)


def _gt_stddev(input):
    a = [0.014, 0.016][0]
    b = [0.014, 0.016][1]
    return ~input * a + input * b


def _gt_length(input):
    a = [1.329, 1.341][0]
    b = [1.329, 1.341][1]

    return ~input * a + input * b


def _per_residue_violation_mask(inputs: Sequence[Tensor], target, mask, temperature):
    output = []

    for input in inputs:
        output = [*output, ((input > target * temperature) * mask)]

    output = torch.max(torch.stack(output, dim=-2), dim=-2)[0]

    x = torch.nn.functional.pad(output, [0, 1])
    y = torch.nn.functional.pad(output, [1, 0])

    return torch.maximum(x, y)


def _bond_length(input, other):
    output = torch.sum((other - input) ** 2, dim=-1)

    return torch.sqrt(output + torch.finfo(input.dtype).eps)


def _loss(input, mask):
    output = torch.sum(input * mask, dim=-1)

    return output / (torch.sum(mask, dim=-1) + torch.finfo(input.dtype).eps)


def _error(input, target):
    return torch.sqrt((input - target) ** 2 + torch.finfo(input.dtype).eps)


def _loss_per_residue(input, target, temperature):
    return torch.nn.functional.relu(input - target * temperature)


def _per_residue_loss_sum(a, b, c):
    """
    Compute a per residue loss (equally distribute the loss to both
    neighbouring residues.
    """
    output = a + b + c

    x = torch.nn.functional.pad(output, [0, 1])
    y = torch.nn.functional.pad(output, [1, 0])

    return 0.5 * (x + y)
