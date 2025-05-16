import dataclasses
import typing

from torch import Tensor

from ._short_string import short_string_to_int

if typing.TYPE_CHECKING:
    from ._residue_array import ResidueArray


def _rename_chains(chain_id: Tensor, mapping: dict[str, str]) -> Tensor:
    masks = {v: short_string_to_int(k) == chain_id for k, v in mapping.items()}

    for c, mask in masks.items():
        chain_id = chain_id.masked_fill(mask, short_string_to_int(c))

    return chain_id


def rename_chains(input: "ResidueArray", mapping: dict[str, str]):
    return dataclasses.replace(
        input, chain_id=_rename_chains(input.chain_id, mapping=mapping)
    )
