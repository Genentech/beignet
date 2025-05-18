import torch

from beignet.structure import ResidueArray
from beignet.structure._frames import (
    backbone_coordinates_to_frames,
    backbone_frames_to_coordinates,
    bbt_to_global_frames,
)


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

    bb_frames, bb_mask = p.backbone_frames
    torsions, torsions_mask = p.torsions

    global_frames, global_frames_mask = bbt_to_global_frames(
        bb_frames, bb_mask, torsions, torsions_mask, p.residue_type
    )

    print(f"{global_frames.shape=}")
