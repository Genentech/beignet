import dataclasses

import optree
import torch
from biotite.database import rcsb
from biotite.structure.io import pdbx

from beignet.structure import ResidueArray


def test_residue_array_atom_array_roundtrip():
    file = pdbx.BinaryCIFFile.read(rcsb.fetch("4cni", "bcif"))
    atom_array = pdbx.get_structure(
        file,
        model=1,
        extra_fields=["b_factor", "occupancy"],
        use_author_fields=True,
    )
    atom_array = atom_array[~atom_array.hetero]
    atom_array = atom_array[~(atom_array.atom_name == "OXT")]

    p = ResidueArray.from_atom_array(atom_array)
    array_roundtrip = p.to_atom_array()
    p_roundtrip = ResidueArray.from_atom_array(array_roundtrip)

    for f in dataclasses.fields(p):
        assert torch.equal(getattr(p, f.name), getattr(p_roundtrip, f.name)), (
            f"{f.name=}"
        )


def test_residue_array_optree():
    file = pdbx.BinaryCIFFile.read(rcsb.fetch("4cni", "bcif"))
    atom_array = pdbx.get_structure(
        file,
        model=1,
        extra_fields=["b_factor", "occupancy"],
        use_author_fields=True,
    )
    atom_array = atom_array[~atom_array.hetero]
    atom_array = atom_array[~(atom_array.atom_name == "OXT")]

    p = ResidueArray.from_atom_array(atom_array)

    p_mapped = optree.tree_map(lambda x: x, p, namespace="beignet")

    for f in dataclasses.fields(p):
        assert torch.equal(getattr(p, f.name), getattr(p_mapped, f.name)), f"{f.name=}"
