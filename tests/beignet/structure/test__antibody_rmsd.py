import dataclasses
from pprint import pprint

from beignet.structure import (
    ResidueArray,
    Rigid,
    antibody_cdr_rmsd,
    swap_symmetric_atom_thin_atoms,
)
from beignet.structure.selectors import ChainSelector


def test_antibody_rmsd_descriptors(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    p = p[ChainSelector(["A", "B"])]
    T = Rigid.rand()

    p_T = dataclasses.replace(
        p,
        atom_thin_xyz=swap_symmetric_atom_thin_atoms(
            p.residue_type, T(p.atom_thin_xyz), p.atom_thin_mask
        )[0],
    )

    result = antibody_cdr_rmsd(p_T, p, heavy_chain="B", light_chain="A")

    pprint(result)
