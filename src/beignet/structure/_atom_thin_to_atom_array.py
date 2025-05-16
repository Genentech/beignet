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
    atom_thin_xyz: Tensor,
    atom_thin_mask: Tensor,
    b_factor: Tensor | None = None,
    occupancy: Tensor | None = None,
) -> AtomArray:
    """Convert atom thin representation to a biotite AtomArray.

    Parameters
    ----------
    residue_type : Tensor
        Integer tensor of shape [L] representing residue types, where L is the sequence length.
        Indices correspond to the positions in STANDARD_RESIDUES.
    chain_id : Tensor
        Tensor of shape [L] containing chain identifiers for each residue.
    author_seq_id : Tensor
        Integer tensor of shape [L] representing residue sequence IDs.
    author_ins_code : Tensor
        Tensor of shape [L] containing insertion codes for each residue.
    atom_thin_xyz : Tensor
        Tensor of shape [L, A, 3] containing atom coordinates, where A is the maximum
        number of atoms per residue in the atom thin representation.
    atom_thin_mask : Tensor
        Boolean tensor of shape [L, A] indicating which atoms are present (True) or absent (False).
    b_factor : Tensor, optional
        Tensor of shape [L, A] containing B-factor values for each atom.
        If None, zeros will be used.
    occupancy : Tensor, optional
        Tensor of shape [L, A] containing occupancy values for each atom.
        If None, ones will be used.

    Returns
    -------
    AtomArray
        A biotite AtomArray containing the protein structure information with
        appropriate annotations (chain_id, res_id, ins_code, res_name, atom_name,
        element, and optionally b_factor and occupancy).
    """
    L, W = atom_thin_mask.nonzero(as_tuple=True)
    n_atoms = L.shape[0]

    residue_type_flat = residue_type[L]
    res_name_flat = numpy.array(
        [AMINO_ACID_1_TO_3[r] for r in STANDARD_RESIDUES] + ["UNK"]
    )[residue_type_flat.cpu().numpy()]

    chain_id_flat = numpy.frombuffer(
        chain_id[L].cpu().numpy().tobytes(), dtype="|S8"
    ).astype(numpy.dtypes.StringDType())

    author_seq_id_flat = author_seq_id[L].cpu().numpy()
    author_ins_code_flat = numpy.frombuffer(
        author_ins_code[L].cpu().numpy().tobytes(), dtype="|S8"
    ).astype(numpy.dtypes.StringDType())

    atom_pos_flat = atom_thin_xyz[L, W].cpu().numpy()

    atom_name = numpy.array(
        [ATOM_THIN_ATOMS[AMINO_ACID_1_TO_3[r]] for r in STANDARD_RESIDUES]
        + [ATOM_THIN_ATOMS["UNK"]]
    )[residue_type[L], W]

    if b_factor is not None:
        b_factor_flat = b_factor[L, W]
    else:
        b_factor_flat = torch.zeros(
            residue_type_flat.shape,
            dtype=atom_thin_xyz.dtype,
            device=atom_thin_xyz.device,
        )

    if occupancy is not None:
        occupancy_flat = occupancy[L, W]
    else:
        occupancy_flat = torch.ones(
            residue_type_flat.shape,
            dtype=atom_thin_xyz.dtype,
            device=atom_thin_xyz.device,
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

    if b_factor is not None:
        atom_array.set_annotation("b_factor", b_factor_flat.cpu().numpy())

    if occupancy is not None:
        atom_array.set_annotation("occupancy", occupancy_flat.cpu().numpy())

    return atom_array
