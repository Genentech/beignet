import typing
from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor

from beignet.constants import CDR_RANGES_AHO

from .._contact_matrix import contact_matrix
from .._short_string import short_string_to_int

if typing.TYPE_CHECKING:
    from .. import ResidueArray


@dataclass
class AllSelector:
    def __call__(self, input: "ResidueArray", **_):
        mask = torch.ones_like(input.chain_id, dtype=torch.bool)
        return mask[..., None]


@dataclass
class ChainSelector:
    which_chains: list[str]

    def __call__(self, input: "ResidueArray", **_):
        mask = torch.zeros_like(input.chain_id, dtype=torch.bool)
        for c in self.which_chains:
            mask = mask | (input.chain_id == short_string_to_int(c))
        return mask[..., None]


@dataclass
class ChainFromAnnotationsSelector:
    key: str

    def __call__(self, input: "ResidueArray", annotations: dict, **_):
        which_chains = annotations.get(self.key, None)
        mask = torch.zeros_like(input.chain_id, dtype=torch.bool)
        if which_chains is not None:
            for c in which_chains:
                mask = mask | (input.chain_id == short_string_to_int(c))
        return mask[..., None]


@dataclass
class ResidueIndexSelector:
    selection: dict[str, list[int]]

    def __call__(self, input: "ResidueArray", **_):
        mask = torch.zeros_like(input.chain_id, dtype=torch.bool)

        for chain, resids in self.selection.items():
            mask = mask | (
                (input.chain_id == short_string_to_int(chain))
                & torch.isin(
                    input.residue_index,
                    torch.as_tensor(resids, device=mask.device, dtype=torch.int64),
                )
            )
        return mask[..., None]


@dataclass
class AuthorSeqIdSelector:
    selection: dict[str, list[int]]

    def __call__(self, input: "ResidueArray", **_):
        mask = torch.zeros_like(input.chain_id, dtype=torch.bool)

        if input.author_seq_id is None:
            raise AssertionError("author_seq_id is None")

        for chain, resids in self.selection.items():
            mask = mask | (
                (input.chain_id == short_string_to_int(chain))
                & torch.isin(
                    input.author_seq_id,
                    torch.as_tensor(resids, device=mask.device, dtype=torch.int64),
                )
            )
        return mask[..., None]


@dataclass
class CDRResidueSelector:
    which_cdrs: list[Literal["H1", "H2", "H3", "H4", "L1", "L2", "L3", "L4"]]
    heavy_chain: str | None = "H"
    light_chain: str | None = "L"
    scheme: Literal["aho"] = "aho"

    def __post_init__(self):
        if not set(self.which_cdrs).issubset(
            {"H1", "H2", "H3", "H4", "L1", "L2", "L3", "L4"}
        ):
            raise KeyError(f"{self.which_cdrs=} not valid")

        if self.scheme not in {"aho"}:
            raise ValueError(f"{self.scheme=} not supported")

    def __call__(self, input: "ResidueArray", **_):
        mask = torch.zeros_like(input.chain_id, dtype=torch.bool)

        for cdr in self.which_cdrs:
            chain = self.heavy_chain if cdr.startswith("H") else self.light_chain
            resids = [i + 1 for i in range(*CDR_RANGES_AHO[cdr])]
            mask = mask | (
                (input.chain_id == short_string_to_int(chain))
                & torch.isin(
                    input.author_seq_id,
                    torch.as_tensor(resids, device=mask.device, dtype=torch.int64),
                )
            )
        return mask[..., None]


@dataclass
class InterfaceResidueSelector:
    selector_A: Callable[["ResidueArray"], Tensor] | Tensor | None = None
    selector_B: Callable[["ResidueArray"], Tensor] | Tensor | None = None
    radius_cutoff: float = 10.0

    def __call__(self, input: "ResidueArray", **_):
        contacts = contact_matrix(
            input,
            selector_A=self.selector_A,
            selector_B=self.selector_B,
            radius_cutoff=self.radius_cutoff,
        )

        is_in_interface = torch.sum(contacts, dim=-1) > 0

        return is_in_interface[..., None]
