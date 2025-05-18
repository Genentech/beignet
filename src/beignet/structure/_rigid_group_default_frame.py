import functools

import torch
from torch import Tensor

from beignet.constants import (
    AMINO_ACID_1_TO_3,
    STANDARD_RESIDUES,
)

from ._rigid import Rigid
from ._rigid_group_atom_coordinates import RIGID_GROUP_ATOM_COORDINATES
from ._torsions import CHI_ANGLES_ATOMS, CHI_ANGLES_MASK


def _make_rigid_transformation_4x4(ex, ey, translation):
    """Create a rigid 4x4 transformation matrix from two axes and transl."""
    # Normalize ex.
    ex = ex / torch.linalg.vector_norm(ex)

    # make ey perpendicular to ex
    ey = ey - torch.dot(ey, ex) * ex
    ey = ey / torch.linalg.vector_norm(ey)

    # compute ez as cross product
    ez = torch.cross(ex, ey, dim=-1)
    m = torch.stack([ex, ey, ez, translation]).transpose(0, 1)
    m = torch.cat([m, torch.as_tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
    return m


@functools.cache
def make_rigid_group_default_frame_4x4() -> Tensor:
    restype_rigid_group_default_frame = torch.zeros([21, 8, 4, 4])

    for i, aa in enumerate(STANDARD_RESIDUES):
        resname = AMINO_ACID_1_TO_3[aa]
        atom_to_xyz = {
            atom_name: torch.as_tensor(xyz)
            for atom_name, _, xyz in RIGID_GROUP_ATOM_COORDINATES[resname]
        }

        # backbone to backbone is the identity transform
        restype_rigid_group_default_frame[i, 0, :, :] = torch.eye(4)

        # pre-omega-frame to backbone (currently dummy identity matrix)
        restype_rigid_group_default_frame[i, 1, :, :] = torch.eye(4)

        # phi-frame to backbone
        mat = _make_rigid_transformation_4x4(
            ex=atom_to_xyz["N"] - atom_to_xyz["CA"],
            ey=torch.as_tensor([1.0, 0.0, 0.0]),
            translation=atom_to_xyz["N"],
        )
        restype_rigid_group_default_frame[i, 2, :, :] = mat

        # psi-frame to backbone
        mat = _make_rigid_transformation_4x4(
            ex=atom_to_xyz["C"] - atom_to_xyz["CA"],
            ey=atom_to_xyz["CA"] - atom_to_xyz["N"],
            translation=atom_to_xyz["C"],
        )
        restype_rigid_group_default_frame[i, 3, :, :] = mat

        # chi1-frame to backbone
        if CHI_ANGLES_MASK[i][0]:
            base_atom_names = CHI_ANGLES_ATOMS[resname][0]
            base_atom_to_xyz = [atom_to_xyz[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_to_xyz[2] - base_atom_to_xyz[1],
                ey=base_atom_to_xyz[0] - base_atom_to_xyz[1],
                translation=base_atom_to_xyz[2],
            )
            restype_rigid_group_default_frame[i, 4, :, :] = mat

        # chi2-frame to chi1-frame
        # chi3-frame to chi2-frame
        # chi4-frame to chi3-frame
        # luckily all rotation axes for the next frame start at (0,0,0) of the
        # previous frame
        for chi_idx in range(1, 4):
            if CHI_ANGLES_MASK[i][chi_idx]:
                axis_end_atom_name = CHI_ANGLES_ATOMS[resname][chi_idx][2]
                axis_end_atom_position = atom_to_xyz[axis_end_atom_name]
                mat = _make_rigid_transformation_4x4(
                    ex=axis_end_atom_position,
                    ey=torch.as_tensor([-1.0, 0.0, 0.0]),
                    translation=axis_end_atom_position,
                )
                restype_rigid_group_default_frame[i, 4 + chi_idx, :, :] = mat

    return restype_rigid_group_default_frame


@functools.cache
def make_bbt_rigid_group_default_frame() -> Rigid:
    # o, chi1, chi2, chi3, chi4
    GROUP_INDICES = (3, 4, 5, 6, 7)

    frames_4x4 = make_rigid_group_default_frame_4x4()[:, GROUP_INDICES]

    t = frames_4x4[:, :, :3, -1]
    r = frames_4x4[:, :, :3, :3]

    return Rigid(t, r)
