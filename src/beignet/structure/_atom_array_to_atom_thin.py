import itertools

import biotite.structure
import numpy
import torch
from biotite.structure import AtomArray

from ._residue_constants import (
    n_atom_thin,
    restype_3to1,
    restype_name_to_atom_thin_names,
    restype_num,
    restype_order,
)


def _mutate_mse_to_met(array: AtomArray) -> AtomArray:
    array = array.copy()
    is_mse = array.res_name == "MSE"
    is_mse_selenium = is_mse & (array.atom_name == "SE")
    array.res_name[is_mse] = "MET"
    array.atom_name[is_mse_selenium] = "SD"
    array.element[is_mse_selenium] = "S"
    array.hetero[is_mse_selenium] = False
    return array

def atom_array_to_atom_thin(
    array: AtomArray,
    mutate_mse_to_met: bool = True,
    use_label_seq_id: bool = False,
    device=None,
    dtype=None,
) -> dict:
    # we only handle amino acids
    aa_mask = numpy.isin(array.res_name, biotite.structure.info.amino_acid_names())
    array = array[aa_mask]

    if mutate_mse_to_met:
        array = _mutate_mse_to_met(array)

    chains = torch.frombuffer(
        bytearray(biotite.structure.get_chains(array).astype("<U2").tobytes()),
        dtype=torch.int64,
    )

    residue_starts = biotite.structure.get_residue_starts(array)
    L = len(residue_starts)

    # residue index of chain start for every residue
    chain_starts = biotite.structure.get_residue_positions(
        array, biotite.structure.get_chain_starts_for(array, residue_starts)
    )

    if use_label_seq_id:
        if "label_seq_id" not in array.get_annotation_categories():
            raise KeyError("label_seq_id not in annotations")
        label_seq_id = torch.from_numpy(array.label_seq_id.astype(int)[residue_starts])
        residue_index = label_seq_id - 1  # adjust to zero based indexing
    else:
        # residue_index goes from [0, ..., C-1] for each chain
        residue_index = torch.tensor(
            biotite.structure.get_residue_positions(array, residue_starts)
            - chain_starts,
            device=device,
        )

    chain_index = torch.tensor(
        biotite.structure.get_chain_positions(array, residue_starts), device=device
    )

    chain_id = chains[chain_index]

    author_seq_id = torch.tensor(array.res_id[residue_starts], device=device)
    author_ins_code = torch.frombuffer(
        bytearray(array.ins_code.astype("<U2").tobytes()), dtype=torch.int64
    )

    residue_type = torch.tensor(
        [
            restype_order.get(
                restype_3to1.get(str(name), "X"),
                restype_num,
            )
            for name in array.res_name[residue_starts]
        ],
        device=device,
    )
    padding_mask = torch.ones_like(residue_type, dtype=bool)

    # residue index of each atom
    residue_positions = torch.tensor(
        biotite.structure.get_residue_positions(array, numpy.arange(len(array))),
        device=device,
    )

    # NOTE no OXT
    known_atom_types = set(
        itertools.chain.from_iterable(restype_name_to_atom_thin_names.values())
    )

    # check which atoms we actually know about
    known_atom_mask = numpy.any(
        numpy.array([array.atom_name == name for name in known_atom_types]),
        axis=0,
    )

    # now filter down to just the known atoms
    array = array[known_atom_mask]
    residue_positions = residue_positions[known_atom_mask]

    atom_thin_idx = torch.tensor(
        [
            restype_name_to_atom_thin_names[res_name].index(str(atom_name))
            for res_name, atom_name in zip(array.res_name, array.atom_name, strict=True)
        ],
        device=device,
    )

    xyz_atom_thin = torch.zeros(L, n_atom_thin, 3, dtype=dtype, device=device)
    xyz_atom_thin.index_put_(
        (residue_positions, atom_thin_idx),
        torch.as_tensor(array.coord, dtype=xyz_atom_thin.dtype, device=device),
    )

    atom_thin_mask = torch.zeros(L, n_atom_thin, dtype=bool, device=device)
    atom_thin_mask.index_put_(
        (residue_positions, atom_thin_idx), torch.tensor(True, device=device)
    )

    if "b_factor" in array.get_annotation_categories():
        b_factors = torch.zeros(L, n_atom_thin, dtype=dtype, device=device)
        b_factors.index_put_(
            (residue_positions, atom_thin_idx),
            torch.tensor(array.b_factor, dtype=b_factors.dtype, device=device),
        )
        b_factors[~atom_thin_mask] = 0.0  # zero out positions we don't want
    else:
        b_factors = None

    if "occupancy" in array.get_annotation_categories():
        occupancies = torch.zeros(L, n_atom_thin, dtype=dtype, device=device)
        occupancies.index_put_(
            (residue_positions, atom_thin_idx),
            torch.tensor(array.occupancy, dtype=occupancies.dtype, device=device),
        )
        occupancies[~atom_thin_mask] = 0.0  # zero out positions we don't want
    else:
        occupancies = None

    return {
        "residue_type": residue_type,
        "padding_mask": padding_mask,
        "residue_index": residue_index,
        "chain_id": chain_id,
        "author_seq_id": author_seq_id,
        "author_ins_code": author_ins_code,
        "xyz_atom_thin": xyz_atom_thin,
        "atom_thin_mask": atom_thin_mask,
        "b_factors": b_factors,
        "occupancies": occupancies,
    }
