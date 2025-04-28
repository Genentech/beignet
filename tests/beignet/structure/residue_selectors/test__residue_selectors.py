from biotite.database import rcsb

from beignet.structure import ResidueArray
from beignet.structure.residue_selectors import (
    ChainSelector,
    ChainSelectorFromAnnotations,
)


def test_chain_selector():
    p = ResidueArray.from_mmcif(rcsb.fetch("7k7r", "cif"), use_seqres=False)
    assert p.chain_id_list == ["A", "B", "C", "D", "E", "F"]

    p = p[ChainSelector(["A"])]
    assert p.chain_id_list == ["A"]


def test_chain_selector_from_annotation():
    p = ResidueArray.from_mmcif(rcsb.fetch("7k7r", "cif"), use_seqres=False)
    assert p.chain_id_list == ["A", "B", "C", "D", "E", "F"]

    p = p[ChainSelectorFromAnnotations("foo")(p, {"foo": "A"})]
    assert p.chain_id_list == ["A"]
