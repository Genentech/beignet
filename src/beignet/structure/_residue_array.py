import io

import fastpdb
import numpy
import optree
import torch
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray
from biotite.structure.io import pdbx as pdbx
from optree.dataclasses import dataclass
from torch import Tensor

from ._atom_array_to_atom_thin import atom_array_to_atom_thin
from ._atom_thin_to_atom_array import atom_thin_to_atom_array
from ._residue_constants import n_atom_thin, restype_order_with_x


def short_string_to_int(input: str):
    """Convert an ascii string with length <= 8 to a uint64 integer."""
    assert input.isascii()
    assert len(input) <= 8
    return int.from_bytes(
        input.ljust(8, "\0").encode("ascii"), byteorder="little", signed=False
    )


def int_to_short_string(input: int):
    assert 0 <= input < 2**64
    """Convert a uint64 integer to an ascii string."""
    return (
        input.to_bytes(length=8, byteorder="little", signed=False)
        .decode("ascii")
        .rstrip("\0")
    )


@dataclass(namespace="beignet")
class ResidueArray:
    residue_type: Tensor
    residue_index: Tensor
    chain_id: Tensor
    padding_mask: Tensor
    xyz_atom_thin: Tensor
    atom_thin_mask: Tensor

    # optional
    author_seq_id: Tensor | None = None
    author_ins_code: Tensor | None = None
    occupancies: Tensor | None = None
    b_factors: Tensor | None = None

    @property
    def chain_id_list(self) -> list[str]:
        return (
            numpy.frombuffer(
                torch.unique_consecutive(self.chain_id[self.padding_mask])
                .cpu()
                .numpy()
                .tobytes(),
                dtype="|S8",
            )
            .astype(numpy.dtypes.StringDType())
            .tolist()
        )

    @classmethod
    def from_sequence(
        cls,
        sequence: str,
        chain_id: str = "A",
        device=None,
        dtype=None,
    ):
        L = len(sequence)
        shape = (L,)

        residue_type = torch.tensor(
            [restype_order_with_x[aa] for aa in sequence], device=device
        )
        residue_index = torch.arange(L, device=device)

        assert len(chain_id) < 8
        assert chain_id.isascii()

        chain_id = torch.full(
            shape,
            short_string_to_int(chain_id),
            device=device,
        )

        padding_mask = torch.ones(shape, device=device, dtype=bool)

        xyz_atom_thin = torch.zeros(
            (*shape, n_atom_thin, 3), device=device, dtype=dtype
        )
        atom_thin_mask = torch.zeros((*shape, n_atom_thin), device=device, dtype=bool)

        author_seq_id = torch.zeros(shape, device=device, dtype=torch.int64)
        author_ins_code = torch.full(shape, ord(" "), device=device, dtype=torch.int64)

        occupancies = torch.ones((*shape, n_atom_thin), device=device, dtype=dtype)
        b_factors = torch.zeros((*shape, n_atom_thin), device=device, dtype=dtype)

        return cls(
            residue_type=residue_type,
            residue_index=residue_index,
            chain_id=chain_id,
            padding_mask=padding_mask,
            xyz_atom_thin=xyz_atom_thin,
            atom_thin_mask=atom_thin_mask,
            author_seq_id=author_seq_id,
            author_ins_code=author_ins_code,
            occupancies=occupancies,
            b_factors=b_factors,
        )

    @classmethod
    def from_chain_sequences(
        cls, chain_sequences: dict[str, str], device=None, dtype=None
    ):
        chains = [
            cls.from_sequence(seq, chain_id=c, device=device, dtype=dtype)
            for i, (c, seq) in enumerate(chain_sequences.items())
        ]

        return optree.tree_map(
            lambda *x: torch.cat([*x], dim=0), *chains, namespace="beignet"
        )

    @classmethod
    def from_atom_array(
        cls,
        array: AtomArray,
        mutate_mse_to_met: bool = True,
        use_label_seq_id: bool = False,
        device=None,
        dtype=None,
    ):
        features = atom_array_to_atom_thin(
            array,
            mutate_mse_to_met=mutate_mse_to_met,
            use_label_seq_id=use_label_seq_id,
            device=device,
            dtype=dtype,
        )
        return cls(**features)

    def to_atom_array(self) -> AtomArray:
        return atom_thin_to_atom_array(
            residue_type=self.residue_type,
            chain_id=self.chain_id,
            author_seq_id=self.author_seq_id,
            author_ins_code=self.author_ins_code,
            xyz_atom_thin=self.xyz_atom_thin,
            atom_thin_mask=self.atom_thin_mask,
            b_factors=self.b_factors,
            occupancies=self.occupancies,
        )

    @classmethod
    def from_pdb(
        cls,
        path,
        model: int = 1,
        mutate_mse_to_met: bool = True,
        device=None,
        dtype=None,
    ):
        file = fastpdb.PDBFile.read(path)
        array = file.get_structure(model=model, extra_fields=["b_factor", "occupancy"])
        return cls.from_atom_array(
            array,
            mutate_mse_to_met=mutate_mse_to_met,
            use_label_seq_id=False,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_mmcif(
        cls,
        path,
        model: int = 1,
        mutate_mse_to_met: bool = True,
        use_seqres: bool = False,
        device=None,
        dtype=None,
    ):
        file = pdbx.CIFFile.read(path)
        return cls._from_cif(
            file,
            model=model,
            mutate_mse_to_met=mutate_mse_to_met,
            use_seqres=use_seqres,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_bcif(
        cls,
        path,
        model: int = 1,
        mutate_mse_to_met: bool = True,
        use_seqres: bool = False,
        device=None,
        dtype=None,
    ):
        file = pdbx.BinaryCIFFile.read(path)
        return cls._from_cif(
            file,
            model=model,
            mutate_mse_to_met=mutate_mse_to_met,
            use_seqres=use_seqres,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def _from_cif(
        cls,
        file: pdbx.CIFFile | pdbx.BinaryCIFFile,
        model: int = 1,
        mutate_mse_to_met: bool = True,
        use_seqres: bool = False,
        device=None,
        dtype=None,
    ):
        array = pdbx.get_structure(
            file,
            model=model,
            extra_fields=["b_factor", "occupancy", "label_seq_id"],
            use_author_fields=True,
        )

        p = cls.from_atom_array(
            array,
            mutate_mse_to_met=mutate_mse_to_met,
            use_label_seq_id=True,
            device=device,
            dtype=dtype,
        )

        if use_seqres:
            # make sure we construct chains in same order as in atom records
            chain_ids = [
                int_to_short_string(c)
                for c in torch.unique_consecutive(p.chain_id).tolist()
            ]
            sequences = pdbx.get_sequence(file)
            p_seqres = ResidueArray.from_chain_sequences(
                {
                    c: str(sequences[c])
                    for c in chain_ids
                    if isinstance(sequences[c], ProteinSequence)
                }
            )

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
                    for c, i in zip(
                        p.chain_id.tolist(), p.residue_index.tolist(), strict=True
                    )
                ],
                device=device,
            )

            # check index consistency
            for field in ["residue_type", "residue_index", "chain_id"]:
                if not torch.equal(
                    getattr(p_seqres, field)[indices], getattr(p, field)
                ):
                    raise RuntimeError(f"seqres indices not consistent for {field}")

            # fill in info from atom records
            optree.tree_map(
                lambda x, y: x.index_put_((indices,), y),
                p_seqres,
                p,
                namespace="beignet",
            )

            return p_seqres
        else:
            return p

    def to_pdb(self, f):
        array = self.to_atom_array()
        file = fastpdb.PDBFile()
        file.set_structure(array)
        file.write(f)

    def to_pdb_string(self) -> str | list[str]:
        buffer = io.StringIO()
        self.to_pdb(buffer)
        return buffer.getvalue()
