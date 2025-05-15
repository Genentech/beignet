import torch

from beignet.constants import CDR_RANGES_AHO
from beignet.structure import ResidueArray, renumber_from_gapped
from beignet.structure.selectors import (
    AndSelector,
    CDRResidueSelector,
    ChainFromAnnotationsSelector,
    ChainSelector,
    PeptideBackboneSelector,
)


def test_chain_selector(structure_7k7r_cif):
    p = ResidueArray.from_mmcif(structure_7k7r_cif, use_seqres=False)
    assert p.chain_id_list == ["A", "B", "C", "D", "E", "F"]

    selected = p[ChainSelector(["A"])]
    assert selected.chain_id_list == ["A"]

    selected = p[ChainSelector(["A", "B"])]
    assert selected.chain_id_list == ["A", "B"]


def test_chain_selector_from_annotation(structure_7k7r_cif):
    p = ResidueArray.from_mmcif(structure_7k7r_cif, use_seqres=False)
    assert p.chain_id_list == ["A", "B", "C", "D", "E", "F"]

    selected = p[ChainFromAnnotationsSelector("foo")(p, {"foo": ["A"]}).any(dim=-1)]
    assert selected.chain_id_list == ["A"]

    selected = p[
        ChainFromAnnotationsSelector("foo")(p, {"foo": ["A", "D"]}).any(dim=-1)
    ]
    assert selected.chain_id_list == ["A", "D"]


def test_renumber_residue_array(structure_7k7r_pdb, gapped_aho_7k7r):
    p = ResidueArray.from_pdb(structure_7k7r_pdb)
    renumbered = renumber_from_gapped(p, gapped_aho_7k7r)

    assert not torch.equal(p.author_seq_id, renumbered.author_seq_id)


def test_cdr_residue_selector(structure_7k7r_cif, gapped_aho_7k7r):
    p = ResidueArray.from_mmcif(structure_7k7r_cif, use_seqres=False)
    assert p.chain_id_list == ["A", "B", "C", "D", "E", "F"]

    p = renumber_from_gapped(p, gapped_aho_7k7r)

    for cdr in [f"H{i}" for i in (1, 2, 3, 4)]:
        expected = gapped_aho_7k7r["B"][slice(*CDR_RANGES_AHO[cdr])].replace("-", "")
        selector = CDRResidueSelector(
            which_cdrs=[cdr], heavy_chain="B", light_chain="A", scheme="aho"
        )
        selected = p[selector]
        assert selected.sequence[selector.heavy_chain] == expected

    for cdr in [f"L{i}" for i in (1, 2, 3, 4)]:
        expected = gapped_aho_7k7r["A"][slice(*CDR_RANGES_AHO[cdr])].replace("-", "")
        selector = CDRResidueSelector(
            which_cdrs=[cdr], heavy_chain="B", light_chain="A", scheme="aho"
        )
        selected = p[selector]
        assert selected.sequence[selector.light_chain] == expected


def test_and_selector(structure_7k7r_cif):
    p = ResidueArray.from_mmcif(structure_7k7r_cif)

    selector1 = ChainSelector(["A"])
    selector2 = PeptideBackboneSelector()

    mask1 = selector1(p)
    mask2 = selector2(p)

    ref = mask1 & mask2

    mask1and2 = AndSelector([selector1, selector2])(p)

    assert torch.equal(mask1and2, ref)
