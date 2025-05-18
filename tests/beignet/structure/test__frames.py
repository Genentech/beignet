import dataclasses

import torch

from beignet.structure import ResidueArray, rmsd
from beignet.structure._frames import (
    backbone_coordinates_to_frames,
    backbone_frames_to_coordinates,
    bbt_to_atom_thin,
)
from beignet.structure.selectors import ChainSelector, PeptideBackboneSelector


def test_backbone_coordinates_to_frames(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)

    bb_frames, bb_mask = backbone_coordinates_to_frames(
        p.atom_thin_xyz[:, :4, :], p.atom_thin_mask[:, :4], p.residue_type
    )

    bb_xyz, bb_xyz_mask = backbone_frames_to_coordinates(
        bb_frames, bb_mask, p.residue_type
    )

    rmsd = torch.sqrt(
        torch.mean(
            torch.sum(
                torch.square(
                    bb_xyz[bb_xyz_mask] - p.atom_thin_xyz[:, :4, :][bb_xyz_mask]
                ),
                dim=-1,
            ),
            dim=-1,
        )
    ).item()

    print(f"{rmsd=:0.2f}")

    assert rmsd < 1.0


def test_bbt(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    p = p[ChainSelector(["A"])]

    bb_frames, bb_mask = p.backbone_frames
    torsions, torsions_mask = p.torsions

    atom_thin_xyz, atom_thin_mask = bbt_to_atom_thin(
        bb_frames, bb_mask, torsions, torsions_mask, p.residue_type
    )

    p_ideal = dataclasses.replace(
        p, atom_thin_xyz=atom_thin_xyz, atom_thin_mask=atom_thin_mask
    )

    rmsd_val = rmsd(p, p_ideal)

    bb_rmsd_val = rmsd(p, p_ideal, selector=PeptideBackboneSelector())

    print(f"{bb_rmsd_val=}")
    print(f"{rmsd_val=}")

    p.to_pdb("7k7r.pdb")
    p_ideal.to_pdb("7k7r_ideal.pdb")
