import dataclasses
import typing

import numpy
import torch
from torch import Tensor

from ._short_string import short_string_to_int

if typing.TYPE_CHECKING:
    from ._residue_array import ResidueArray


def _gapped_domain_to_numbering(
    gapped: str,
    full: str,
    pre_gap: int = 0,
    post_gap: int = 0,
) -> list[tuple[int, str]]:
    domain = gapped.replace("-", "")
    start = full.find(domain)
    if start == -1:
        raise RuntimeError("gapped not found in full")

    pre = [i - pre_gap for i in range(-start, 0)]
    mid = [i for i, r in enumerate(gapped) if r != "-"]
    post = [
        i + post_gap
        for i in range(len(gapped), len(gapped) + (len(full) - len(domain) - start))
    ]

    out = pre + mid + post

    if not len(out) == len(full):
        raise AssertionError(f"{len(out)=} != {len(full)}")

    # convert to one indexed
    # NOTE no insertion codes
    return [(i + 1, "") for i in out]


def _renumber(
    numbering: dict[str, list[tuple[int, str]]],
    chain_id: Tensor,
    author_seq_id: Tensor,
    author_ins_code: Tensor,
) -> tuple[Tensor, Tensor]:
    for chain, chain_numbering in numbering.items():
        chain_mask = chain_id == short_string_to_int(chain)
        chain_length = chain_mask.sum().item()

        if chain_length == 0:
            raise KeyError(f"{chain=} not found")

        if chain_length != len(chain_numbering):
            raise AssertionError(
                f"{chain=}: {chain_length=} != {len(chain_numbering)=}"
            )

        ids, ins = list(zip(*chain_numbering, strict=True))
        ids = torch.as_tensor(ids, dtype=torch.int64, device=author_seq_id.device)
        ins = torch.frombuffer(
            bytearray(numpy.array(ins).astype("|S8").tobytes()), dtype=torch.int64
        ).to(device=author_ins_code.device)

        author_seq_id = torch.masked_scatter(author_seq_id, chain_mask, ids)
        author_ins_code = torch.masked_scatter(author_ins_code, chain_mask, ins)

    return author_seq_id, author_ins_code


def _renumber_from_gapped(
    gapped: dict[str, str],
    sequence: dict[str, str],
    chain_id: Tensor,
    author_seq_id: Tensor,
    author_ins_code: Tensor,
    pre_gap: int = 0,
    post_gap: int = 0,
) -> tuple[Tensor, Tensor]:
    numbering = {
        k: _gapped_domain_to_numbering(
            gapped=v, full=sequence[k], pre_gap=pre_gap, post_gap=post_gap
        )
        for k, v in gapped.items()
    }

    author_seq_id, author_ins_code = _renumber(
        numbering=numbering,
        chain_id=chain_id,
        author_seq_id=author_seq_id,
        author_ins_code=author_ins_code,
    )

    return author_seq_id, author_ins_code


def renumber(
    input: "ResidueArray", numbering: dict[str, list[tuple[int, str]]]
) -> "ResidueArray":
    author_seq_id, author_ins_code = _renumber(
        numbering,
        chain_id=input.chain_id,
        author_seq_id=input.author_seq_id,
        author_ins_code=input.author_ins_code,
    )

    return dataclasses.replace(
        input,
        author_seq_id=author_seq_id,
        author_ins_code=author_ins_code,
    )


def renumber_from_gapped(
    input: "ResidueArray",
    gapped: dict[str, str],
    pre_gap: int = 0,
    post_gap: int = 0,
) -> "ResidueArray":
    author_seq_id, author_ins_code = _renumber_from_gapped(
        gapped=gapped,
        sequence=input.sequence,
        chain_id=input.chain_id,
        author_seq_id=input.author_seq_id,
        author_ins_code=input.author_ins_code,
        pre_gap=pre_gap,
        post_gap=post_gap,
    )

    return dataclasses.replace(
        input,
        author_seq_id=author_seq_id,
        author_ins_code=author_ins_code,
    )
