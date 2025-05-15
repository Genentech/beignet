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
                ).bool()
                if atom_name in v
                else torch.zeros(n_atom_thin, device=device, dtype=torch.bool)
            )
            for v in ATOM_THIN_ATOMS.values()
        ]
    )


@dataclass
class AtomNameSelector:
    which_atoms: list[str]

    def __call__(self, input: "ResidueArray") -> Tensor:
        mask = torch.zeros(
            len(ATOM_THIN_ATOMS),
            n_atom_thin,
            device=input.residue_type.device,
            dtype=torch.bool,
        )

        for atom_name in self.which_atoms:
            mask = mask | _atom_name_mask(atom_name, device=input.residue_type.device)

        mask = mask[input.residue_type]

        return mask


@dataclass
class AlphaCarbonSelector:
    def __call__(self, input: "ResidueArray") -> Tensor:
        return AtomNameSelector(["CA"])(input)


@dataclass
class PeptideBackboneSelector:
    include_oxygen: bool = False

    def __call__(self, input: "ResidueArray") -> Tensor:
        if self.include_oxygen:
            atom_names = ["CA", "C", "N", "O"]
        else:
            atom_names = ["CA", "C", "N"]

        return AtomNameSelector(atom_names)(input)
