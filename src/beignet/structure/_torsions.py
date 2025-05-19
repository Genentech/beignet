import functools

import einops
import torch
from torch import Tensor

from beignet import dihedral_angle
from beignet.constants import (
    AMINO_ACID_1_TO_3,
    ATOM_THIN_ATOMS,
    STANDARD_RESIDUES,
)

# fmt: off
# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
CHI_ANGLES_ATOMS = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [ ["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "NE"], ["CG", "CD", "NE", "CZ"], ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]], # skip chi2: ["CA", "CB", "SG", "HG"]
    "GLN": [ ["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"], ],
    "GLU": [ ["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"], ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [ ["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "CE"], ["CG", "CD", "CE", "NZ"], ],
    "MET": [ ["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "SD"], ["CB", "CG", "SD", "CE"], ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"], ["CA", "CB", "OG", "HG"]],
    "THR": [["N", "CA", "CB", "OG1"], ["CA", "CB", "OG1", "HG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"], ["CE1", "CZ", "OH", "HH"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}
# fmt: on

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
CHI_ANGLES_MASK = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.
CHI_PI_PERIODIC = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 1.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 1.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 1.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]


@functools.cache
def make_chi_angle_atom_indices_tensor() -> Tensor:
    chi_angle_atom_thin_indices = [
        [
            [
                ATOM_THIN_ATOMS[AMINO_ACID_1_TO_3[aa]].index(atom_name)
                for atom_name in atoms
            ]
            for atoms in CHI_ANGLES_ATOMS[AMINO_ACID_1_TO_3[aa]]
        ]
        + ([[0, 0, 0, 0]] * (4 - len(CHI_ANGLES_ATOMS[AMINO_ACID_1_TO_3[aa]])))
        for aa in STANDARD_RESIDUES
    ]

    # [20, 4, 4]
    return torch.as_tensor(chi_angle_atom_thin_indices)


def atom_thin_to_oxygen_torsion(
    atom_thin_xyz: Tensor,
    atom_thin_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    reference_atom_indices = (0, 1, 2, 3)  # [N, Ca, C, O]
    coords = atom_thin_xyz[..., reference_atom_indices, :]  # [..., 4, 3]
    mask = atom_thin_mask[..., reference_atom_indices]

    psi_o = dihedral_angle(coords) + torch.pi
    psi_o_mask = torch.all(mask, dim=-1)

    return psi_o, psi_o_mask


def atom_thin_to_chi_torsion(
    atom_thin_xyz: Tensor, atom_thin_mask: Tensor, residue_type: Tensor, chi_index: int
) -> tuple[Tensor, Tensor]:
    if not 0 <= chi_index < 4:
        raise ValueError(f"{chi_index=} must be between 0 and 3")
    device = residue_type.device

    chi_angles_mask = (torch.as_tensor(CHI_ANGLES_MASK, device=device) == 1.0)[
        residue_type, chi_index
    ]

    chi_angles_atom_indices = make_chi_angle_atom_indices_tensor().to(device)
    chi_angles_atom_indices = chi_angles_atom_indices[residue_type, chi_index]

    coords = torch.gather(
        atom_thin_xyz,
        dim=-2,
        index=einops.repeat(chi_angles_atom_indices, "... -> ... 3"),
    )
    mask = torch.gather(atom_thin_mask, dim=-1, index=chi_angles_atom_indices)

    psi = dihedral_angle(coords)
    psi_mask = torch.all(mask, dim=-1) & chi_angles_mask

    return psi, psi_mask


def atom_thin_to_torsions(
    atom_thin_xyz: Tensor, atom_thin_mask: Tensor, residue_type: Tensor
) -> tuple[Tensor, Tensor]:
    psi_o, psi_o_mask = atom_thin_to_oxygen_torsion(atom_thin_xyz, atom_thin_mask)
    chi1, chi1_mask = atom_thin_to_chi_torsion(
        atom_thin_xyz, atom_thin_mask, residue_type, 0
    )
    chi2, chi2_mask = atom_thin_to_chi_torsion(
        atom_thin_xyz, atom_thin_mask, residue_type, 1
    )
    chi3, chi3_mask = atom_thin_to_chi_torsion(
        atom_thin_xyz, atom_thin_mask, residue_type, 2
    )
    chi4, chi4_mask = atom_thin_to_chi_torsion(
        atom_thin_xyz, atom_thin_mask, residue_type, 3
    )

    torsion = torch.stack([psi_o, chi1, chi2, chi3, chi4], dim=-1)
    torsion_mask = torch.stack(
        [psi_o_mask, chi1_mask, chi2_mask, chi3_mask, chi4_mask], dim=-1
    )

    return torsion, torsion_mask
