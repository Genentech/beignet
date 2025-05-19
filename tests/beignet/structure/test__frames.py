import dataclasses

import torch

from beignet.structure import ResidueArray, rmsd
from beignet.structure._frames import (
    bbt_to_atom_thin,
)
from beignet.structure.selectors import ChainSelector, PeptideBackboneSelector


def test_bbt_rmsd(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    p = p[ChainSelector(["A", "B"])]

    bb_frames, bb_mask = p.backbone_frames
    torsions, torsions_mask = p.torsions

    atom_thin_xyz, atom_thin_mask = bbt_to_atom_thin(
        bb_frames, bb_mask, torsions, torsions_mask, p.residue_type
    )

    assert torch.equal(atom_thin_mask, p.atom_thin_mask)

    p_ideal = dataclasses.replace(
        p, atom_thin_xyz=atom_thin_xyz, atom_thin_mask=atom_thin_mask
    )

    bb_rmsd_val = rmsd(p, p_ideal, selector=PeptideBackboneSelector()).item()
    print(f"{bb_rmsd_val=:0.2e}")
    assert bb_rmsd_val < 0.1

    rmsd_val = rmsd(p, p_ideal).item()
    print(f"{rmsd_val=:0.2e}")
    assert rmsd_val < 0.1


def test_bbt_roundtrip_rmsd(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    p = p[ChainSelector(["A", "B"])]

    bb_frames, bb_mask = p.backbone_frames
    torsions, torsions_mask = p.torsions

    atom_thin_xyz, atom_thin_mask = bbt_to_atom_thin(
        bb_frames, bb_mask, torsions, torsions_mask, p.residue_type
    )

    assert torch.equal(atom_thin_mask, p.atom_thin_mask)

    p_ideal = dataclasses.replace(
        p, atom_thin_xyz=atom_thin_xyz, atom_thin_mask=atom_thin_mask
    )

    bb_frames_ideal, bb_mask_ideal = p_ideal.backbone_frames
    torsions_ideal, torsions_mask_ideal = p.torsions

    atom_thin_xyz_roundtrip, atom_thin_mask_roundtrip = bbt_to_atom_thin(
        bb_frames_ideal,
        bb_mask_ideal,
        torsions_ideal,
        torsions_mask_ideal,
        p.residue_type,
    )

    p_roundtrip = dataclasses.replace(
        p,
        atom_thin_xyz=atom_thin_xyz_roundtrip,
        atom_thin_mask=atom_thin_mask_roundtrip,
    )

    bb_rmsd_val = rmsd(p_ideal, p_roundtrip, selector=PeptideBackboneSelector()).item()
    print(f"{bb_rmsd_val=:0.2e}")
    assert bb_rmsd_val < 1e-4

    rmsd_val = rmsd(p_ideal, p_roundtrip).item()
    print(f"{rmsd_val=:0.2e}")
    assert rmsd_val < 1e-4
