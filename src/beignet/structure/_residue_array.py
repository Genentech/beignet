import dataclasses
import functools
import io
import operator
from typing import Callable, Literal

import einops
import fastpdb
import numpy
import optree
import torch
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray
from biotite.structure.io import pdbx as pdbx
from optree.dataclasses import dataclass
from torch import Tensor

from beignet import pad_to_target_length
from beignet.constants import ATOM_THIN_ATOMS, STANDARD_RESIDUES

from ._atom_array_to_atom_thin import atom_array_to_atom_thin
from ._atom_thin_to_atom_array import atom_thin_to_atom_array
from ._backbone_coordinates_to_dihedrals import backbone_coordinates_to_dihedrals
from ._rename_symmetric_atoms import rename_symmetric_atoms
from ._rigid import Rigid
from ._short_string import int_to_short_string, short_string_to_int

restypes_with_x = STANDARD_RESIDUES + ["X"]
restype_order_with_x = {r: i for i, r in enumerate(restypes_with_x)}
n_atom_thin = len(ATOM_THIN_ATOMS["ALA"])

HANDLED_FUNCTIONS = {}


def _atom_name_mask(atom_name: str, device=None) -> Tensor:
    return torch.stack(
        [
            (
                torch.nn.functional.one_hot(
                    torch.as_tensor(v.index(atom_name), device=device), n_atom_thin
                )
                if atom_name in v
                else torch.zeros(n_atom_thin, device=device, dtype=torch.int64)
            )
            for v in ATOM_THIN_ATOMS.values()
        ]
    )


def _gapped_domain_to_numbering(
    gapped: str,
    full: str,
    pre_gap: int = 0,
    post_gap: int = 0,
) -> list[tuple[int, str]]:
    domain = gapped.replace("-", "")
    start = full.find(domain)
    if start == -1:
        raise RuntimeError("gapped not found in full")

    pre = [i - pre_gap for i in range(-start, 0)]
    mid = [i for i, r in enumerate(gapped) if r != "-"]
    post = [
        i + post_gap
        for i in range(len(gapped), len(gapped) + (len(full) - len(domain) - start))
    ]

    out = pre + mid + post

    if not len(out) == len(full):
        raise AssertionError(f"{len(out)=} != {len(full)}")

    # convert to one indexed
    # NOTE no insertion codes
    return [(i + 1, "") for i in out]


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@dataclass(namespace="beignet")
class ResidueArray:
    residue_type: Tensor
    residue_index: Tensor
    chain_id: Tensor
    padding_mask: Tensor
    atom_thin_xyz: Tensor
    atom_thin_mask: Tensor

    # optional
    author_seq_id: Tensor | None = None
    author_ins_code: Tensor | None = None
    occupancy: Tensor | None = None
    b_factor: Tensor | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.residue_type.shape

    @property
    def L(self) -> int:
        return self.residue_type.shape[-1]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def chain_id_list(self) -> list[str]:
        if self.ndim != 1:
            raise RuntimeError(
                f"ResidueArray.chain_id_list only supported for ndim == 1 {self.ndim=}"
            )

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

    @property
    def sequence(self) -> dict[str, str]:
        if self.ndim != 1:
            raise RuntimeError(
                f"ResidueArray.chain_sequences only supported for ndim == 1 {self.ndim=}"
            )

        return {
            c: str.join(
                "",
                [
                    restypes_with_x[i]
                    for i in self.residue_type[
                        self.chain_id == short_string_to_int(c)
                    ].tolist()
                ],
            )
            for c in self.chain_id_list
        }

    @property
    def backbone_coordinates(self) -> tuple[Tensor, Tensor]:
        return self.atom_thin_xyz[..., :3, :], self.atom_thin_mask[..., :3]

    @property
    def backbone_dihedrals(self) -> tuple[Tensor, Tensor]:
        coords, mask = self.backbone_coordinates
        return backbone_coordinates_to_dihedrals(
            backbone_coordinates=coords,
            mask=mask,
            residue_index=self.residue_index,
            chain_id=self.chain_id,
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

        atom_thin_xyz = torch.zeros(
            (*shape, n_atom_thin, 3), device=device, dtype=dtype
        )
        atom_thin_mask = torch.zeros((*shape, n_atom_thin), device=device, dtype=bool)

        author_seq_id = torch.zeros(shape, device=device, dtype=torch.int64)
        author_ins_code = torch.full(shape, ord(" "), device=device, dtype=torch.int64)

        occupancy = torch.ones((*shape, n_atom_thin), device=device, dtype=dtype)
        b_factor = torch.zeros((*shape, n_atom_thin), device=device, dtype=dtype)

        return cls(
            residue_type=residue_type,
            residue_index=residue_index,
            chain_id=chain_id,
            padding_mask=padding_mask,
            atom_thin_xyz=atom_thin_xyz,
            atom_thin_mask=atom_thin_mask,
            author_seq_id=author_seq_id,
            author_ins_code=author_ins_code,
            occupancy=occupancy,
            b_factor=b_factor,
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
        if self.ndim != 1:
            raise RuntimeError(
                f"ResidueArray.to_atom_array only supported for ndim == 1 {self.ndim=}"
            )

        return atom_thin_to_atom_array(
            residue_type=self.residue_type,
            chain_id=self.chain_id,
            author_seq_id=self.author_seq_id,
            author_ins_code=self.author_ins_code,
            atom_thin_xyz=self.atom_thin_xyz,
            atom_thin_mask=self.atom_thin_mask,
            b_factor=self.b_factor,
            occupancy=self.occupancy,
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

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, ResidueArray)) for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __getitem__(self, key):
        if callable(key):
            return self[key(self)]
        return optree.tree_map(
            lambda x: operator.getitem(x, key), self, namespace="beignet"
        )

    def to_pdb(self, f):
        if self.ndim != 1:
            raise RuntimeError(
                f"ResidueArray.to_atom_array only supported for ndim == 1 {self.ndim=}"
            )

        array = self.to_atom_array()
        file = fastpdb.PDBFile()
        file.set_structure(array)
        file.write(f)

    def to_pdb_string(self) -> str:
        if self.ndim != 1:
            raise RuntimeError(
                f"ResidueArray.to_atom_array only supported for ndim == 1 {self.ndim=}"
            )

        buffer = io.StringIO()
        self.to_pdb(buffer)
        return buffer.getvalue()

    def pad_to_target_length(self, target_length: int, dim: int = 0):
        return optree.tree_map(
            lambda x: pad_to_target_length(x, target_length=target_length, dim=dim),
            self,
            namespace="beignet",
        )

    def to(self, dtype=None, device=None):
        return optree.tree_map(
            lambda x: x.to(
                dtype=dtype if torch.is_floating_point(x) else None,
                device=device,
            ),
            self,
            namespace="beignet",
        )

    def renumber(self, numbering: dict[str, list[tuple[int, str]]]):
        """Renumber ResidueArray.

        Parameters
        ----------
        numbering: dict[str, list[tuple[int, str]]]
            Dictionary mapping from chain ids to lists of
            (index, insertion_code) tuples.

        Returns
        -------
        ResidueArray
            New object with `author_seq_id` and `author_ins_code` fields updated
        """
        if self.ndim != 1:
            raise RuntimeError(
                f"ResidueArray.renumber only supported for ndim == 1 {self.ndim=}"
            )

        author_seq_id = self.author_seq_id
        author_ins_code = self.author_ins_code

        chain_ids = self.chain_id_list
        for chain, chain_numbering in numbering.items():
            if chain not in chain_ids:
                raise KeyError(f"{chain=} not in structure {chain_ids=}")

            chain_mask = self.chain_id == short_string_to_int(chain)
            chain_length = chain_mask.sum().item()

            if chain_length != len(chain_numbering):
                raise AssertionError(
                    f"{chain=}: {chain_length=} != {len(chain_numbering)=}"
                )

            ids, ins = list(zip(*chain_numbering, strict=True))
            ids = torch.as_tensor(
                ids, dtype=torch.int64, device=self.author_seq_id.device
            )
            ins = torch.frombuffer(
                bytearray(numpy.array(ins).astype("|S8").tobytes()), dtype=torch.int64
            ).to(device=self.author_ins_code.device)

            author_seq_id = torch.masked_scatter(author_seq_id, chain_mask, ids)
            author_ins_code = torch.masked_scatter(author_ins_code, chain_mask, ins)

        return dataclasses.replace(
            self, author_seq_id=author_seq_id, author_ins_code=author_ins_code
        )

    def renumber_from_gapped_domain(
        self, gapped: dict[str, str], pre_gap: int = 0, post_gap: int = 0
    ):
        sequence = self.sequence
        numbering = {
            k: _gapped_domain_to_numbering(
                gapped=v, full=sequence[k], pre_gap=pre_gap, post_gap=post_gap
            )
            for k, v in gapped.items()
        }
        return self.renumber(numbering)

    def _rmsd(
        self,
        target: "ResidueArray",
        residue_selector: Callable[["ResidueArray"], Tensor] | None = None,
        atom_selector: Literal["c_alpha", "all"] = "all",
        align: bool = False,
        optimize_ambiguous_atoms: bool = False,
        **residue_selector_kwargs,
    ):
        match atom_selector:
            case "c_alpha":
                atom_mask = _atom_name_mask("CA", device=self.residue_type.device)[
                    self.residue_type
                ]
            case "all":
                atom_mask = torch.ones_like(self.atom_thin_mask)
            case _:
                raise AssertionError(f"{atom_selector=} not supported")

        if residue_selector is not None:
            residue_mask = residue_selector(self, **residue_selector_kwargs)
        else:
            residue_mask = torch.ones_like(self.padding_mask)

        if optimize_ambiguous_atoms:
            atom_thin_xyz = rename_symmetric_atoms(
                self.residue_type,
                self.atom_thin_xyz,
                self.atom_thin_mask,
                target.atom_thin_xyz,
            )
        else:
            atom_thin_xyz = self.atom_thin_xyz

        atom_thin_mask = self.atom_thin_mask

        x = einops.rearrange(atom_thin_xyz, "... l n d -> ... (l n) d")
        x_mask = einops.rearrange(atom_thin_mask, "... l n -> ... (l n)")

        y = einops.rearrange(target.atom_thin_xyz, "... l n d -> ... (l n) d")
        y_mask = einops.rearrange(target.atom_thin_mask, "... l n -> ... (l n)")

        atom_mask = einops.rearrange(atom_mask, "... l n -> ... (l n)")

        weights = x_mask & y_mask
        weights = weights & einops.repeat(
            residue_mask, f"... l -> ... (l {n_atom_thin})"
        )
        weights = weights & einops.repeat(
            self.padding_mask, f"... l -> ... (l {n_atom_thin})"
        )
        weights = weights & atom_mask

        if align:
            T = Rigid.kabsch(y, x, weights=weights, keepdim=True)
            x = T(x)
        else:
            T = None

        delta2 = torch.square(x - y)

        a = einops.reduce(delta2 * weights[..., None], "... n d -> ...", "sum")
        b = einops.reduce(weights, "... n -> ...", "sum")

        rmsd = torch.sqrt(a / b)

        return rmsd, T

    def rmsd(
        self,
        target: "ResidueArray",
        residue_selector: Callable[["ResidueArray"], Tensor] | None = None,
        atom_selector: Literal["c_alpha", "all"] = "all",
        align: bool = False,
        optimize_ambiguous_atoms: bool = False,
        **residue_selector_kwargs,
    ) -> Tensor:
        rmsd, _ = self._rmsd(
            target=target,
            residue_selector=residue_selector,
            atom_selector=atom_selector,
            align=align,
            optimize_ambiguous_atoms=optimize_ambiguous_atoms,
            **residue_selector_kwargs,
        )

        return rmsd

    def kabsch(
        self,
        target: "ResidueArray",
        residue_selector: Callable[["ResidueArray"], Tensor] | None = None,
        atom_selector: Literal["c_alpha", "all"] = "all",
        optimize_ambiguous_atoms: bool = False,
        **residue_selector_kwargs,
    ) -> Rigid:
        _, T = self._rmsd(
            target=target,
            residue_selector=residue_selector,
            atom_selector=atom_selector,
            align=True,
            optimize_ambiguous_atoms=optimize_ambiguous_atoms,
            **residue_selector_kwargs,
        )

        return T


@implements(torch.cat)
def cat(input, dim=0):
    return optree.tree_map(
        lambda *x: torch.cat([*x], dim=dim), *input, namespace="beignet"
    )


@implements(torch.stack)
def stack(input, dim=0):
    return optree.tree_map(
        lambda *x: torch.stack([*x], dim=dim), *input, namespace="beignet"
    )


@implements(torch.unbind)
def unbind(input, dim=0):
    return optree.tree_transpose_map(
        lambda x: torch.unbind(x, dim=dim), input, namespace="beignet"
    )


@implements(torch.unsqueeze)
def unsqueeze(input, dim: int):
    if dim < 0:
        dim = input.shape - dim
    return optree.tree_map(
        lambda x: torch.unsqueeze(x, dim=dim), input, namespace="beignet"
    )


@implements(torch.squeeze)
def squeeze(input, dim: int):
    if dim < 0:
        dim = input.shape - dim
    return optree.tree_map(
        lambda x: torch.squeeze(x, dim=dim), input, namespace="beignet"
    )
