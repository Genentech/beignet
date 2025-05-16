import dataclasses

import torch

from beignet.structure import (
    ResidueArray,
    Rigid,
    superimpose,
    swap_symmetric_atom_thin_atoms,
)
from beignet.structure.selectors import AlphaCarbonSelector, ChainSelector


def test_superimpose(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    T = Rigid.rand(1)

    p_T = dataclasses.replace(p, atom_thin_xyz=T(p.atom_thin_xyz))

    _, T_kabsch, rmsd = superimpose(p_T, p)

    print(f"{rmsd=}")

    assert rmsd.item() < 1e-4

    torch.testing.assert_close(T_kabsch.t, T.t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(T_kabsch.r, T.r, atol=1e-4, rtol=1e-4)


def test_superimpose_with_selector(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    T = Rigid.rand(1)

    p_T = dataclasses.replace(p, atom_thin_xyz=T(p.atom_thin_xyz))

    _, T_kabsch, rmsd = superimpose(p_T, p, selector=AlphaCarbonSelector())

    print(f"{rmsd=}")

    assert rmsd.item() < 1e-4

    torch.testing.assert_close(T_kabsch.t, T.t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(T_kabsch.r, T.r, atol=1e-4, rtol=1e-4)


def test_superimpose_with_rename_symmetric_atoms(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    T = Rigid.rand()

    p_T = dataclasses.replace(
        p,
        atom_thin_xyz=swap_symmetric_atom_thin_atoms(
            p.residue_type, T(p.atom_thin_xyz), p.atom_thin_mask
        )[0],
    )

    _, _, rmsd = superimpose(p_T, p, rename_symmetric_atoms=False)
    print(f"{rmsd=}")

    assert rmsd.item() > 0.1

    _, _, rmsd = superimpose(p_T, p, rename_symmetric_atoms=True)
    print(f"{rmsd=}")

    assert rmsd.item() < 1e-4


def test_superimpose_batched(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    T = Rigid.rand()
    p_T = dataclasses.replace(
        p,
        atom_thin_xyz=swap_symmetric_atom_thin_atoms(
            p.residue_type, T(p.atom_thin_xyz), p.atom_thin_mask
        )[0],
    )

    batch = torch.stack(
        [p[ChainSelector([c])].pad_to_target_length(256) for c in p.chain_id_list]
    )

    batch_T = torch.stack(
        [p_T[ChainSelector([c])].pad_to_target_length(256) for c in p.chain_id_list]
    )

    _, _, rmsd = superimpose(batch_T, batch)

    print(f"{rmsd=}")
    assert rmsd.shape == (6,)
    assert (rmsd < 1e-4).all()
