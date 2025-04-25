import numpy
import torch
from biotite.structure import AtomArray
from torch import Tensor

from beignet.constants import AMINO_ACID_1_TO_3, ATOM_THIN_ATOMS, STANDARD_RESIDUES

restypes_with_x = STANDARD_RESIDUES + ["X"]


def atom_thin_to_atom_array(
    residue_type: Tensor,
    chain_id: Tensor,
    author_seq_id: Tensor,
    author_ins_code: Tensor,
    xyz_atom_thin: Tensor,
    atom_thin_mask: Tensor,
    b_factors: Tensor | None = None,
    occupancies: Tensor | None = None,
) -> AtomArray:
    L, W = atom_thin_mask.nonzero(as_tuple=True)
    n_atoms = L.shape[0]

    residue_type_flat = residue_type[L]
    res_name_flat = numpy.array(
        [AMINO_ACID_1_TO_3[r] for r in STANDARD_RESIDUES] + ["UNK"]
    )[residue_type_flat.cpu().numpy()]

    chain_id_flat = numpy.frombuffer(
        chain_id[L].cpu().numpy().tobytes(), dtype="|S8"
    ).astype(numpy.dtypes.StringDType())

    author_seq_id_flat = author_seq_id[L]
    author_ins_code_flat = numpy.frombuffer(
        author_ins_code[L].cpu().numpy().tobytes(), dtype="|S8"
    ).astype(numpy.dtypes.StringDType())

    atom_pos_flat = xyz_atom_thin[L, W]

    atom_name = numpy.array(
        [ATOM_THIN_ATOMS[AMINO_ACID_1_TO_3[r]] for r in STANDARD_RESIDUES]
        + [ATOM_THIN_ATOMS["UNK"]]
    )[residue_type[L], W]

    if b_factors is not None:
        b_factors_flat = b_factors[L, W]
    else:
        b_factors_flat = torch.zeros(
            residue_type_flat.shape,
            dtype=xyz_atom_thin.dtype,
            device=xyz_atom_thin.device,
        )

    if occupancies is not None:
        occupancies_flat = occupancies[L, W]
    else:
        occupancies_flat = torch.ones(
            residue_type_flat.shape,
            dtype=xyz_atom_thin.dtype,
            device=xyz_atom_thin.device,
        )

    atom_array = AtomArray(n_atoms)
    atom_array.coord[:] = atom_pos_flat
    atom_array.chain_id[:] = chain_id_flat
    atom_array.res_id[:] = author_seq_id_flat
    atom_array.ins_code[:] = author_ins_code_flat
    atom_array.res_name[:] = res_name_flat
    atom_array.hetero[:] = False
    atom_array.atom_name[:] = atom_name

    # Protein supports only C, N, O, S, this works
    atom_array.element[:] = atom_array.atom_name.astype("<U1")

    if b_factors is not None:
        atom_array.set_annotation("b_factor", b_factors_flat.cpu().numpy())

    if occupancies is not None:
        atom_array.set_annotation("occupancy", occupancies_flat.cpu().numpy())

    return atom_array
