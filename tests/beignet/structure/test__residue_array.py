import dataclasses
import io

import biotite.structure
import optree
import torch
from biotite.structure.io import pdbx

from beignet.constants import ATOM_THIN_ATOMS
from beignet.structure import ResidueArray


def test_atom_thin_atoms():
    n_atom_thin = len(ATOM_THIN_ATOMS["ALA"])
    for k, v in ATOM_THIN_ATOMS.items():
        assert len(v) == n_atom_thin, f"{k=}"


def test_residue_array_from_cif(
    structure_7k7r_pdb, structure_7k7r_cif, structure_7k7r_bcif
):
    p0 = ResidueArray.from_pdb(structure_7k7r_pdb)
    p1 = ResidueArray.from_mmcif(structure_7k7r_cif)
    p2 = ResidueArray.from_bcif(structure_7k7r_bcif)

    assert torch.equal(p0.atom_thin_xyz, p1.atom_thin_xyz)
    assert torch.equal(p0.atom_thin_xyz, p2.atom_thin_xyz)


def test_residue_array_chain_id_list(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    assert p.chain_id_list == ["A", "B", "C", "D", "E", "F"]


def test_residue_array_from_cif_with_seqres(structure_7k7r_cif):
    p = ResidueArray.from_mmcif(structure_7k7r_cif, use_seqres=False)

    structure_7k7r_cif.seek(0)  # reset stream
    p_seqres = ResidueArray.from_mmcif(structure_7k7r_cif, use_seqres=True)

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


def test_residue_array_atom_array_roundtrip(structure_7k7r_bcif):
    file = pdbx.BinaryCIFFile.read(structure_7k7r_bcif)
    atom_array = pdbx.get_structure(
        file,
        model=1,
        extra_fields=["b_factor", "occupancy"],
        use_author_fields=True,
    )
    atom_array = atom_array[~atom_array.hetero]

    p = ResidueArray.from_atom_array(atom_array)
    array_roundtrip = p.to_atom_array()
    p_roundtrip = ResidueArray.from_atom_array(array_roundtrip)

    for f in dataclasses.fields(p):
        assert torch.equal(getattr(p, f.name), getattr(p_roundtrip, f.name)), (
            f"{f.name=}"
        )


def test_residue_array_optree(structure_7k7r_bcif):
    file = pdbx.BinaryCIFFile.read(structure_7k7r_bcif)
    atom_array = pdbx.get_structure(
        file,
        model=1,
        extra_fields=["b_factor", "occupancy"],
        use_author_fields=True,
    )
    atom_array = atom_array[~atom_array.hetero]

    p = ResidueArray.from_atom_array(atom_array)

    leaves, spec = optree.tree_flatten(p, namespace="beignet")
    assert len(leaves) == len(dataclasses.fields(p))

    p_mapped = optree.tree_unflatten(spec, leaves)

    for f in dataclasses.fields(p):
        assert torch.equal(getattr(p, f.name), getattr(p_mapped, f.name)), f"{f.name=}"


def test_residue_array_pdb_roundtrip(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    pdb_string = p.to_pdb_string()
    p_roundtrip = ResidueArray.from_pdb(io.StringIO(pdb_string))

    for f in dataclasses.fields(p):
        assert torch.equal(getattr(p, f.name), getattr(p_roundtrip, f.name)), (
            f"{f.name=}"
        )


def test_residue_array_pdb_roundtrip_with_ins_code(structure_1s78_pdb):
    p = ResidueArray.from_pdb(structure_1s78_pdb)
    pdb_string = p.to_pdb_string()
    p_roundtrip = ResidueArray.from_pdb(io.StringIO(pdb_string))

    for f in dataclasses.fields(p):
        assert torch.equal(getattr(p, f.name), getattr(p_roundtrip, f.name)), (
            f"{f.name=}"
        )


def test_residue_array_from_sequence():
    p = ResidueArray.from_chain_sequences({"A": "AAAA", "B": "LLLLL"})
    assert p.residue_type.shape == (9,)


def test_residue_array_cat(structure_7k7r_pdb):
    p0 = ResidueArray.from_pdb(structure_7k7r_pdb)

    assert p0.shape == (891,)

    p = torch.cat([p0, p0], dim=0)

    assert p.shape == (2 * 891,)


def test_residue_array_stack(structure_7k7r_pdb):
    p0 = ResidueArray.from_pdb(structure_7k7r_pdb)

    assert p0.shape == (891,)

    p = torch.stack([p0, p0], dim=0)

    assert p.shape == (
        2,
        891,
    )


def test_residue_array_unbind(structure_7k7r_pdb):
    p0 = ResidueArray.from_pdb(structure_7k7r_pdb)

    assert p0.shape == (891,)

    p = torch.stack([p0, p0], dim=0)

    a, b = torch.unbind(p, dim=0)

    for f in dataclasses.fields(p0):
        assert torch.equal(getattr(p0, f.name), getattr(a, f.name)), f"{f.name=}"
        assert torch.equal(getattr(p0, f.name), getattr(b, f.name)), f"{f.name=}"


def test_residue_array_slice(structure_7k7r_pdb):
    p0 = ResidueArray.from_pdb(structure_7k7r_pdb)
    a = p0[:100]
    b = p0[100:]

    assert a.shape == (100,)
    assert b.shape == (891 - 100,)

    p1 = torch.cat([a, b], dim=0)

    for f in dataclasses.fields(p0):
        assert torch.equal(getattr(p0, f.name), getattr(p1, f.name)), f"{f.name=}"


def test_residue_array_to_backbone_dihedrals(structure_7k7r_bcif):
    file = pdbx.BinaryCIFFile.read(structure_7k7r_bcif)
    atom_array = pdbx.get_structure(
        file,
        model=1,
        extra_fields=["b_factor", "occupancy"],
        use_author_fields=True,
    )
    atom_array = atom_array[~atom_array.hetero]
    atom_array = atom_array[atom_array.chain_id == "A"]

    phi_ref, psi_ref, omega_ref = biotite.structure.dihedral_backbone(atom_array)

    p = ResidueArray.from_atom_array(atom_array)

    dihedrals, _ = p.backbone_dihedrals

    phi, psi, omega = torch.unbind(dihedrals, dim=-1)

    torch.testing.assert_close(phi, torch.from_numpy(phi_ref), equal_nan=True)
    torch.testing.assert_close(psi, torch.from_numpy(psi_ref), equal_nan=True)
    torch.testing.assert_close(omega, torch.from_numpy(omega_ref), equal_nan=True)


def test_residue_array_to_chain_sequences(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    L = p.shape[0]
    seq = p.sequence

    assert len(seq.keys()) == 6
    assert sum(len(v) for v in seq.values()) == L


def test_residue_array_type_conversion(structure_7k7r_pdb):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    p = p.to(torch.float64)

    assert p.residue_type.dtype == torch.int64
    assert p.atom_thin_xyz.dtype == torch.float64
    assert p.occupancy.dtype == torch.float64
    assert p.b_factor.dtype == torch.float64
