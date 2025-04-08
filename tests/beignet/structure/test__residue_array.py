import dataclasses
import io

import optree
import torch
from biotite.database import rcsb
from biotite.structure.io import pdbx

from beignet.structure import ResidueArray


def test_residue_array_from_cif():
    p0 = ResidueArray.from_pdb(rcsb.fetch("1A8O", "pdb"))
    p1 = ResidueArray.from_mmcif(rcsb.fetch("1A8O", "cif"))
    p2 = ResidueArray.from_bcif(rcsb.fetch("1A8O", "bcif"))

    assert torch.equal(p0.xyz_atom_thin, p1.xyz_atom_thin)
    assert torch.equal(p0.xyz_atom_thin, p2.xyz_atom_thin)


def test_residue_array_chain_id_list():
    p = ResidueArray.from_pdb(rcsb.fetch("7k7r", "pdb"))
    assert p.chain_id_list == ["A", "B", "C", "D", "E", "F"]


def test_residue_array_from_cif_with_seqres():
    p = ResidueArray.from_mmcif(rcsb.fetch("7k7r", "cif"), use_seqres=False)
    p_seqres = ResidueArray.from_mmcif(rcsb.fetch("7k7r", "cif"), use_seqres=True)

    assert p.residue_type.shape == (891,)
    assert p_seqres.residue_type.shape == (926,)

    index_mapping = {
        (c, i): j
        for j, (c, i) in enumerate(
            zip(
                p_seqres.chain_id.tolist(),
                p_seqres.residue_index.tolist(),
                strict=True,
            )
        )
    }
    indices = torch.tensor(
        [
            index_mapping[(c, i)]
            for c, i in zip(p.chain_id.tolist(), p.residue_index.tolist(), strict=True)
        ],
    )

    # check index consistency
    for field in ["residue_type", "residue_index", "chain_id"]:
        if not torch.equal(getattr(p_seqres, field)[indices], getattr(p, field)):
            raise RuntimeError(f"seqres indices not consistent for {field}")


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

    leaves, spec = optree.tree_flatten(p, namespace="beignet")
    assert len(leaves) == len(dataclasses.fields(p))

    p_mapped = optree.tree_unflatten(spec, leaves)

    for f in dataclasses.fields(p):
        assert torch.equal(getattr(p, f.name), getattr(p_mapped, f.name)), f"{f.name=}"


def test_residue_array_pdb_roundtrip():
    p = ResidueArray.from_pdb(rcsb.fetch("4cni", "pdb"))
    pdb_string = p.to_pdb_string()
    p_roundtrip = ResidueArray.from_pdb(io.StringIO(pdb_string))

    for f in dataclasses.fields(p):
        assert torch.equal(getattr(p, f.name), getattr(p_roundtrip, f.name)), (
            f"{f.name=}"
        )


def test_residue_array_pdb_roundtrip_with_ins_code():
    p = ResidueArray.from_pdb(rcsb.fetch("1s78", "pdb"))
    pdb_string = p.to_pdb_string()
    p_roundtrip = ResidueArray.from_pdb(io.StringIO(pdb_string))

    for f in dataclasses.fields(p):
        assert torch.equal(getattr(p, f.name), getattr(p_roundtrip, f.name)), (
            f"{f.name=}"
        )


def test_residue_array_from_sequence():
    p = ResidueArray.from_chain_sequences({"A": "AAAA", "B": "LLLLL"})
    assert p.residue_type.shape == (9,)


def test_residue_array_cat():
    p0 = ResidueArray.from_pdb(rcsb.fetch("7k7r", "pdb"))

    assert p0.shape == (891,)

    p = torch.cat([p0, p0], dim=0)

    assert p.shape == (2 * 891,)


def test_residue_array_stack():
    p0 = ResidueArray.from_pdb(rcsb.fetch("7k7r", "pdb"))

    assert p0.shape == (891,)

    p = torch.stack([p0, p0], dim=0)

    assert p.shape == (
        2,
        891,
    )


def test_residue_array_unbind():
    p0 = ResidueArray.from_pdb(rcsb.fetch("7k7r", "pdb"))

    assert p0.shape == (891,)

    p = torch.stack([p0, p0], dim=0)

    a, b = torch.unbind(p, dim=0)

    for f in dataclasses.fields(p0):
        assert torch.equal(getattr(p0, f.name), getattr(a, f.name)), f"{f.name=}"
        assert torch.equal(getattr(p0, f.name), getattr(b, f.name)), f"{f.name=}"


def test_residue_array_slice():
    p0 = ResidueArray.from_pdb(rcsb.fetch("7k7r", "pdb"))
    a = p0[:100]
    b = p0[100:]

    assert a.shape == (100,)
    assert b.shape == (891 - 100,)

    p1 = torch.cat([a, b], dim=0)

    for f in dataclasses.fields(p0):
        assert torch.equal(getattr(p0, f.name), getattr(p1, f.name)), f"{f.name=}"
