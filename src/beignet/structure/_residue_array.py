from biotite.structure import AtomArray
from optree.dataclasses import dataclass
from torch import Tensor

from ._atom_array_to_atom_thin import atom_array_to_atom_thin
from ._atom_thin_to_atom_array import atom_thin_to_atom_array


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
