import dataclasses

from beignet.structure import ResidueArray, Rigid
from beignet.structure._rename_symmetric_atoms import _swap_symmetric_atom_thin_atoms
from beignet.structure.metrics import AntibodyRMSDDescriptors
from beignet.structure.residue_selectors import ChainSelector


def test_antibody_rmsd_descriptors(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    p = p[ChainSelector(["A", "B"])]
    T = Rigid.rand()

    p_T = dataclasses.replace(
        p,
        atom_thin_xyz=_swap_symmetric_atom_thin_atoms(
            p.residue_type, T(p.atom_thin_xyz), p.atom_thin_mask
        )[0],
    )

    descriptors = AntibodyRMSDDescriptors.from_residue_array(
        p_T, p, heavy_chain="B", light_chain="A"
    )

    print(descriptors)
