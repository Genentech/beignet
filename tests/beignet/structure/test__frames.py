import dataclasses

import pytest
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

    assert torch.equal(p_ideal.atom_thin_mask, p_roundtrip.atom_thin_mask)

    bb_rmsd_val = rmsd(p_ideal, p_roundtrip, selector=PeptideBackboneSelector()).item()
    print(f"{bb_rmsd_val=:0.2e}")
    assert bb_rmsd_val < 1e-4

    rmsd_val = rmsd(p_ideal, p_roundtrip).item()
    print(f"{rmsd_val=:0.2e}")
    assert rmsd_val < 1e-4


@pytest.mark.parametrize("chi_index", [1, 2, 3, 4])
def test_chi_mask(structure_7k7r_pdb, chi_index):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    p = p[ChainSelector(["A", "B"])]

    bb_frames, bb_mask = p.backbone_frames
    torsions, torsions_mask = p.torsions

    # check that chi1-4 are present before we mask
    assert torsions_mask[:, (1, 2, 3, 4)].any(dim=0).all().item()

    # mask out chi_i
    torsions_mask[:, chi_index] = False

    atom_thin_xyz, atom_thin_mask = bbt_to_atom_thin(
        bb_frames, bb_mask, torsions, torsions_mask, p.residue_type
    )

    p_masked = dataclasses.replace(
        p, atom_thin_xyz=atom_thin_xyz, atom_thin_mask=atom_thin_mask
    )
    for i in (1, 2, 3, 4):
        if i >= chi_index:
            assert not p_masked.torsions[1][:, i].any()
