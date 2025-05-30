import functools
import itertools

import einops
import torch
from torch import Tensor

from beignet.constants import AMINO_ACID_3_TO_1, ATOM_THIN_ATOMS, STANDARD_RESIDUES
from beignet.constants._amino_acid_1_to_3 import AMINO_ACID_1_TO_3

from ._rigid import Rigid
from ._rigid_group_atom_coordinates import RIGID_GROUP_ATOM_COORDINATES
from ._rigid_group_default_frame import make_bbt_rigid_group_default_frame


@functools.cache
def make_frame_to_xyz_dict() -> dict:
    """Create dictionary with default atom coordinates data.

    Returns
    -------
    frame_to_atoms

    key is tuple of (residue_index, frame_index).
    value is tuple of (atom_names, atom_coordinates).
    frame_index is in [0, 6) and corresponds to bb, o, chi1, chi2, chi3, chi4.
    atom_coordinates have Shape (NAtom,3) where NAtom is the number of atoms in the frame.
    """
    # bb, o, chi1, chi2, chi3, chi4
    GROUP_INDICES = [
        0,
        3,
        4,
        5,
        6,
        7,
    ]  # refers to group_index in rigid_group_atom_coordinates
    out = {}
    for resname, resdata in RIGID_GROUP_ATOM_COORDINATES.items():
        residue_index = STANDARD_RESIDUES.index(AMINO_ACID_3_TO_1[resname])

        for k, g in itertools.groupby(resdata, key=lambda x: x[1]):
            g = list(g)
            frame_index = GROUP_INDICES.index(k)
            atom_names = [x[0] for x in g]
            default_atom_coordinates = [x[2] for x in g]
            k = (residue_index, frame_index)
            out[k] = (atom_names, torch.as_tensor(default_atom_coordinates))
    return out


@functools.cache
def make_frame_to_xyz_tensors() -> tuple[Tensor, Tensor, Tensor]:
    """Create tensor holding default atom coordinates for each residue/frame.

    Returns
    -------
    xyz: Tensor
        Shape [N_RESIDUE_TYPES, N_FRAMES, MAX_N_ATOMS_PER_FRAME, 3] == [20, 6, 8, 3]
    mask: Tensor
        Shape [N_RESIDUE_TYPES, N_FRAMES, MAX_N_ATOMS_PER_FRAME] == [20, 6, 8]
    atom_thin_index: Tensor
        Shape [N_RESIDUE_TYPES, N_FRAMES, MAX_N_ATOMS_PER_FRAME] == [20, 6, 8]
    """
    frame_to_xyz = make_frame_to_xyz_dict()
    N_RESIDUE_TYPES = 20
    N_FRAMES = 6  # bb, 0, chi1, chi2, chi3, chi4
    MAX_N_ATOMS_PER_FRAME = max(v[1].shape[0] for _, v in frame_to_xyz.items())
    xyz = torch.zeros(N_RESIDUE_TYPES, N_FRAMES, MAX_N_ATOMS_PER_FRAME, 3)
    mask = torch.zeros(
        N_RESIDUE_TYPES, N_FRAMES, MAX_N_ATOMS_PER_FRAME, dtype=torch.bool
    )
    atom_thin_index = torch.zeros(
        N_RESIDUE_TYPES, N_FRAMES, MAX_N_ATOMS_PER_FRAME, dtype=torch.int64
    )
    for (i, j), (
        atom_names,
        default_xyz,
    ) in frame_to_xyz.items():
        resname = AMINO_ACID_1_TO_3[STANDARD_RESIDUES[i]]
        n_atoms = default_xyz.shape[0]
        xyz[i, j, :n_atoms, :] = default_xyz
        mask[i, j, :n_atoms] = True
        atom_thin_index[i, j, :n_atoms] = torch.tensor(
            [ATOM_THIN_ATOMS[resname].index(n) for n in atom_names]
        )

    return xyz, mask, atom_thin_index


def atom_thin_to_backbone_frames(
    atom_thin_xyz: Tensor, atom_thin_mask: Tensor, residue_type: Tensor
):
    # [20, 6, 8]
    default_xyz, default_mask, atom_thin_index = make_frame_to_xyz_tensors()

    # [L, 4]
    bb_atom_thin_index = atom_thin_index.to(residue_type.device)[residue_type, 0, :4]

    bb_xyz = torch.gather(
        atom_thin_xyz, dim=-2, index=einops.repeat(bb_atom_thin_index, "... -> ... 3")
    )
    mask = torch.gather(atom_thin_mask, dim=-1, index=bb_atom_thin_index)

    bb_default_xyz = default_xyz.to(residue_type.device)[residue_type, 0, :4, :]
    mask = mask & default_mask.to(residue_type.device)[residue_type, 0, :4]

    bb_mask = mask.sum(dim=-1) >= 3
    bb_frames = Rigid.kabsch(bb_xyz, bb_default_xyz, weights=mask, keepdim=False)

    return bb_frames, bb_mask


def _rotation_x(phi: Tensor) -> Tensor:
    """Generate a rotation matrix for a rotation by angle phi around the x-axis."""
    cos = torch.cos(phi)
    sin = torch.sin(phi)

    ones = torch.ones_like(cos)
    zeros = torch.zeros_like(cos)

    R = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos, sin], dim=-1),
            torch.stack([zeros, -sin, cos], dim=-1),
        ],
        dim=-1,
    )
    return R


def _rigid_rotation_x(phi: Tensor) -> Rigid:
    r = _rotation_x(phi)
    t = torch.zeros(
        (*phi.shape, 3),
        device=phi.device,
        dtype=phi.dtype,
        requires_grad=phi.requires_grad,
    )
    return Rigid(t, r)


def bbt_to_global_frames(
    bb_frames: Rigid,
    bb_mask: Tensor,
    torsions: Tensor,
    torsions_mask: Tensor,
    residue_type: Tensor,
    default_frames: Rigid | None = None,
) -> tuple[Rigid, Tensor]:
    if default_frames is None:
        default_frames = make_bbt_rigid_group_default_frame().to(
            dtype=bb_frames.t.dtype, device=bb_frames.t.device
        )
        default_frames = default_frames[residue_type]

    psi_o, chi1, chi2, chi3, chi4 = torsions.unbind(dim=-1)
    psi_o_mask, chi1_mask, chi2_mask, chi3_mask, chi4_mask = torsions_mask.unbind(
        dim=-1
    )

    T_o_local = _rigid_rotation_x(psi_o)
    T_chi1_local = _rigid_rotation_x(chi1)
    T_chi2_local = _rigid_rotation_x(chi2)
    T_chi3_local = _rigid_rotation_x(chi3)
    T_chi4_local = _rigid_rotation_x(chi4)

    o_mask = bb_mask & psi_o_mask
    chi1_mask = bb_mask & chi1_mask
    chi2_mask = chi1_mask & chi2_mask
    chi3_mask = chi2_mask & chi3_mask
    chi4_mask = chi3_mask & chi4_mask

    otobb, chi1tobb, chi2tochi1, chi3tochi2, chi4tochi3 = torch.unbind(
        default_frames, dim=-1
    )

    T_o = bb_frames.compose(otobb).compose(T_o_local)
    T_chi1 = bb_frames.compose(chi1tobb).compose(T_chi1_local)
    T_chi2 = T_chi1.compose(chi2tochi1).compose(T_chi2_local)
    T_chi3 = T_chi2.compose(chi3tochi2).compose(T_chi3_local)
    T_chi4 = T_chi3.compose(chi4tochi3).compose(T_chi4_local)

    global_frames = torch.stack(
        [bb_frames, T_o, T_chi1, T_chi2, T_chi3, T_chi4], dim=-1
    )
    global_frames_mask = torch.stack(
        [bb_mask, o_mask, chi1_mask, chi2_mask, chi3_mask, chi4_mask], dim=-1
    )

    return global_frames, global_frames_mask


def bbt_to_atom_thin(
    bb_frames: Rigid,
    bb_mask: Tensor,
    torsions: Tensor,
    torsions_mask: Tensor,
    residue_type: Tensor,
    default_frames: Rigid | None = None,
) -> tuple[Tensor, Tensor]:
    global_frames, global_frames_mask = bbt_to_global_frames(
        bb_frames=bb_frames,
        bb_mask=bb_mask,
        torsions=torsions,
        torsions_mask=torsions_mask,
        residue_type=residue_type,
        default_frames=default_frames,
    )

    default_xyz, default_mask, atom_thin_index = make_frame_to_xyz_tensors()

    # [L, 6, 8, 3]
    default_xyz = default_xyz.to(residue_type.device)[residue_type]
    default_mask = default_mask.to(residue_type.device)[residue_type]

    # [L, 6, 8, 3]
    xyz = torch.unsqueeze(global_frames, dim=-1)(default_xyz)
    xyz_mask = default_mask & global_frames_mask[..., None]

    # [N, 3]
    xyz = xyz[xyz_mask]

    # [L, 6, 8]
    atom_thin_index = atom_thin_index.to(residue_type.device)[residue_type]

    # [N,]
    atom_thin_index = atom_thin_index[xyz_mask]

    n_atom_thin = len(ATOM_THIN_ATOMS["ALA"])

    atom_thin_xyz = torch.zeros(
        *residue_type.shape, n_atom_thin, 3, device=residue_type.device
    )

    atom_thin_mask = torch.zeros(
        *residue_type.shape, n_atom_thin, device=residue_type.device, dtype=torch.bool
    )

    indices = torch.nonzero(xyz_mask, as_tuple=True)

    atom_thin_xyz = torch.index_put(
        atom_thin_xyz, (*indices[:-2], atom_thin_index), xyz
    )
    atom_thin_mask = torch.index_put(
        atom_thin_mask,
        (*indices[:-2], atom_thin_index),
        torch.as_tensor(True, device=residue_type.device),
    )

    return atom_thin_xyz, atom_thin_mask
