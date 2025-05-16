import biotite.structure
import numpy
import torch
from biotite.structure import AtomArray
from torch import Tensor

from beignet.constants import (
    AMINO_ACID_3_TO_1,
    ATOM_THIN_ATOMS,
    STANDARD_RESIDUES,
)

restype_order = {r: i for i, r in enumerate(STANDARD_RESIDUES)}
restype_num = len(STANDARD_RESIDUES)
n_atom_thin = len(ATOM_THIN_ATOMS["ALA"])


def _mutate_mse_to_met(array: AtomArray) -> AtomArray:
    array = array.copy()
    is_mse = array.res_name == "MSE"
    is_mse_selenium = is_mse & (array.atom_name == "SE")
    array.res_name[is_mse] = "MET"
    array.atom_name[is_mse_selenium] = "SD"
    array.element[is_mse_selenium] = "S"
    array.hetero[is_mse_selenium] = False
    return array


def _mutate_sec_to_cys(array: AtomArray) -> AtomArray:
    array = array.copy()
    is_sec = array.res_name == "SEC"
    is_sec_selenium = is_sec & (array.atom_name == "SE")
    array.res_name[is_sec] = "CYS"
    array.atom_name[is_sec_selenium] = "SG"
    array.element[is_sec_selenium] = "S"
    array.hetero[is_sec_selenium] = False
    return array


def _selenium_to_sulfur(array: AtomArray) -> AtomArray:
    array = _mutate_mse_to_met(array)
    array = _mutate_sec_to_cys(array)
    return array


def atom_array_to_atom_thin(
    array: AtomArray,
    selenium_to_sulfur: bool = True,
    use_label_seq_id: bool = False,
    device=None,
    dtype=None,
) -> dict[str, Tensor]:
    """Convert a biotite `AtomArray` into dict of torch tensors in "atom thin" format.

    Parameters
    ----------
    array: AtomArray
        input data
    selenium_to_sulfur: bool = True
        If True, mutate MSE to MET and SEC to CYS.
    use_label_seq_id: bool = False
        If True use `label_seq_id` annotation of `array` to set `residue_index`
    device = None
        device for torch tensors
    dtype = None
        dtype for floating point data

    Returns
    -------
    dict[str, Tensor | None]
        A dictionary of shape [L, *] torch tensors where L is the
        number of amino acid residues in the input array.
        Keys:
        - "residue_type"
        - "padding_mask"
        - "residue_index"
        - "chain_id"
        - "author_seq_id"
        - "author_ins_code"
        - "atom_thin_xyz"
        - "atom_thin_mask"
        - "b_factor"
        - "occupancy"
    """
    # we only handle amino acids
    aa_mask = numpy.isin(array.res_name, biotite.structure.info.amino_acid_names())
    array = array[aa_mask]

    if selenium_to_sulfur:
        array = _selenium_to_sulfur(array)

    chains = torch.frombuffer(
        bytearray(biotite.structure.get_chains(array).astype("|S8").tobytes()),
        dtype=torch.int64,
    ).to(device=device)

    residue_starts = biotite.structure.get_residue_starts(array)
    L = len(residue_starts)

    # residue index of chain start for every residue
    chain_starts = biotite.structure.get_residue_positions(
        array, biotite.structure.get_chain_starts_for(array, residue_starts)
    )

    if use_label_seq_id:
        if "label_seq_id" not in array.get_annotation_categories():
            raise KeyError("label_seq_id not in annotations")
        label_seq_id = torch.from_numpy(
            array.label_seq_id.astype(int)[residue_starts]
        ).to(device=device)
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
        bytearray(array.ins_code[residue_starts].astype("|S8").tobytes()),
        dtype=torch.int64,
    ).to(device=device)

    residue_type = torch.tensor(
        [
            restype_order.get(
                AMINO_ACID_3_TO_1.get(str(name), "X"),
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

    # check which atoms we actually know about
    known_atom_mask = numpy.any(
        numpy.array(
            [
                numpy.isin(array.atom_name, v) & (array.res_name == k)
                for k, v in ATOM_THIN_ATOMS.items()
            ]
        ),
        axis=0,
    )

    # now filter down to just the known atoms
    array = array[known_atom_mask]
    residue_positions = residue_positions[known_atom_mask]

    atom_thin_idx = torch.tensor(
        [
            ATOM_THIN_ATOMS[res_name].index(str(atom_name))
            for res_name, atom_name in zip(array.res_name, array.atom_name, strict=True)
        ],
        device=device,
    )

    atom_thin_xyz = torch.zeros(L, n_atom_thin, 3, dtype=dtype, device=device)
    atom_thin_xyz.index_put_(
        (residue_positions, atom_thin_idx),
        torch.as_tensor(array.coord, dtype=atom_thin_xyz.dtype, device=device),
    )

    atom_thin_mask = torch.zeros(L, n_atom_thin, dtype=bool, device=device)
    atom_thin_mask.index_put_(
        (residue_positions, atom_thin_idx), torch.tensor(True, device=device)
    )

    if "b_factor" in array.get_annotation_categories():
        b_factor = torch.zeros(L, n_atom_thin, dtype=dtype, device=device)
        b_factor.index_put_(
            (residue_positions, atom_thin_idx),
            torch.tensor(array.b_factor, dtype=b_factor.dtype, device=device),
        )
        b_factor[~atom_thin_mask] = 0.0  # zero out positions we don't want
    else:
        b_factor = None

    if "occupancy" in array.get_annotation_categories():
        occupancy = torch.zeros(L, n_atom_thin, dtype=dtype, device=device)
        occupancy.index_put_(
            (residue_positions, atom_thin_idx),
            torch.tensor(array.occupancy, dtype=occupancy.dtype, device=device),
        )
        occupancy[~atom_thin_mask] = 0.0  # zero out positions we don't want
    else:
        occupancy = None

    return {
        "residue_type": residue_type,
        "padding_mask": padding_mask,
        "residue_index": residue_index,
        "chain_id": chain_id,
        "author_seq_id": author_seq_id,
        "author_ins_code": author_ins_code,
        "atom_thin_xyz": atom_thin_xyz,
        "atom_thin_mask": atom_thin_mask,
        "b_factor": b_factor,
        "occupancy": occupancy,
    }
