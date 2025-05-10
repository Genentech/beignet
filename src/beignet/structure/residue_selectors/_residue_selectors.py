from dataclasses import dataclass
from typing import Literal

import torch

from beignet.constants import CDR_RANGES_AHO

from .._residue_array import ResidueArray
from .._short_string import short_string_to_int


@dataclass
class AllSelector:
    def __call__(self, input: ResidueArray, **_):
        mask = torch.ones_like(input.chain_id, dtype=torch.bool)
        return mask


@dataclass
class ChainSelector:
    which_chains: list[str]

    def __call__(self, input: ResidueArray, **_):
        mask = torch.zeros_like(input.chain_id, dtype=torch.bool)
        for c in self.which_chains:
            mask = mask | (input.chain_id == short_string_to_int(c))
        return mask


@dataclass
class ChainSelectorFromAnnotations:
    key: str

    def __call__(self, input: ResidueArray, annotations: dict, **_):
        which_chains = annotations.get(self.key, None)
        mask = torch.zeros_like(input.chain_id, dtype=torch.bool)
        if which_chains is not None:
            for c in which_chains:
                mask = mask | (input.chain_id == short_string_to_int(c))
        return mask


@dataclass
class ResidueIndexSelector:
    selection: dict[str, list[int]]

    def __call__(self, input: ResidueArray, **_):
        mask = torch.zeros_like(input.chain_id, dtype=torch.bool)

        for chain, resids in self.selection.items():
            mask = mask | (
                (input.chain_id == short_string_to_int(chain))
                & torch.isin(
                    input.author_seq_id,
                    torch.as_tensor(resids, device=mask.device, dtype=torch.int64),
                )
            )
        return mask


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

    def __call__(self, input: ResidueArray, **_):
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
        return mask
