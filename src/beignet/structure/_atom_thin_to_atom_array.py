import biotite.structure
import numpy
import torch
from biotite.structure import Atom, AtomArray
from torch import Tensor

from ._residue_constants import (
    restype_1to3,
    restype_name_to_atom_thin_names,
    restypes_with_x,
)


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

    residue_type_flat = residue_type[L]
    chain_id_flat = numpy.frombuffer(
        chain_id[L].cpu().numpy().tobytes(), dtype="|S8"
    ).astype(numpy.dtypes.StringDType())

    author_seq_id_flat = author_seq_id[L]
    author_ins_code_flat = numpy.frombuffer(
        author_ins_code[L].cpu().numpy().tobytes(), dtype="|S8"
    ).astype(numpy.dtypes.StringDType())

    atom_pos_flat = xyz_atom_thin[L, W]

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

    # FIXME this loop is slow
    atoms = []
    for (
        residue_type_i,
        chain_id_i,
        author_seq_id_i,
        author_ins_code_i,
        atom_pos_i,
        atom_idx_i,
        b_factor_i,
        occupancy_i,
    ) in zip(
        residue_type_flat.cpu().numpy(),
        chain_id_flat,
        author_seq_id_flat.cpu().numpy(),
        author_ins_code_flat,
        atom_pos_flat.cpu().numpy(),
        W.cpu().numpy(),
        b_factors_flat.cpu().numpy(),
        occupancies_flat.cpu().numpy(),
        strict=True,
    ):
        res_name_1 = restypes_with_x[residue_type_i]
        res_name_3 = restype_1to3.get(res_name_1, "UNK")
        atom_name = restype_name_to_atom_thin_names[res_name_3][atom_idx_i]
        element = atom_name[0]  # Protein supports only C, N, O, S, this works.

        atoms.append(
            Atom(
                atom_pos_i,
                chain_id=chain_id_i,
                res_id=author_seq_id_i.item(),
                ins_code=author_ins_code_i,
                res_name=res_name_3,
                hetero=False,
                atom_name=atom_name,
                element=element,
                b_factor=b_factor_i.item(),
                occupancy=occupancy_i.item(),
            )
        )

    array = biotite.structure.array(atoms)
    return array
