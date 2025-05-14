import typing
from dataclasses import dataclass

import torch
from torch import Tensor

from beignet.constants import ATOM_THIN_ATOMS

if typing.TYPE_CHECKING:
    from .. import ResidueArray

n_atom_thin = len(ATOM_THIN_ATOMS["ALA"])


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


@dataclass
class AlphaCarbonSelector:
    def __call__(self, input: "ResidueArray") -> Tensor:
        mask = _atom_name_mask("CA", device=input.residue_type.device)[
            input.residue_type
        ]
        return mask


@dataclass
class ProteinBackboneSelector:
    def __call__(self, input: "ResidueArray") -> Tensor:
        mask = _atom_name_mask("CA", device=input.residue_type.device)
        mask = mask | _atom_name_mask("C", device=input.residue_type.device)
        mask = mask | _atom_name_mask("N", device=input.residue_type.device)
        mask = mask[input.residue_type]
        return mask
