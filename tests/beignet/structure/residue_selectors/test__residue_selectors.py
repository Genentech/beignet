import torch
from biotite.database import rcsb

from beignet.constants import CDR_RANGES_AHO
from beignet.structure import ResidueArray
from beignet.structure.residue_selectors import (
    CDRResidueSelector,
    ChainSelector,
    ChainSelectorFromAnnotations,
)

GAPPED_AHO_7K7R = {
    "A": "DVVLTQSPLSLPVILGQPASISCRSS--QSLVYSD-GRTYLNWFQQRPGQSPRRLIYK--------ISKRDSGVPERFSGSGSG--TDFTLEISRVEAEDVGIYYCMQGSH-----------------------WPVTFGQGTKVEIKR",
    "B": "-VQLVES-GGGLVKPGGSLRLSCVSSG-FTFSN-----YWMSWVRQAPGGGLEWVANINQD---GSEKYYVDSVKGRFTSSRDNTKNSLFLQLNSLRAEDTGIYYCTRDPP-----------------------YFDNWGQGTLVTVSS",
    "D": "DVVLTQSPLSLPVILGQPASISCRSS--QSLVYSD-GRTYLNWFQQRPGQSPRRLIYK--------ISKRDSGVPERFSGSGSG--TDFTLEISRVEAEDVGIYYCMQGSH-----------------------WPVTFGQGTKVEIKR",
    "E": "QVQLVES-GGGLVKPGGSLRLSCVSSG-FTFSN-----YWMSWVRQAPGGGLEWVANINQD---GSEKYYVDSVKGRFTSSRDNTKNSLFLQLNSLRAEDTGIYYCTRDPP-----------------------YFDNWGQGTLVTVSS",
}


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


def test_renumber_residue_array():
    p = ResidueArray.from_pdb(rcsb.fetch("7k7r", "pdb"))
    renumbered = p.renumber_from_gapped_domain(GAPPED_AHO_7K7R)

    assert not torch.equal(p.author_seq_id, renumbered.author_seq_id)


def test_cdr_residue_selector():
    p = ResidueArray.from_mmcif(rcsb.fetch("7k7r", "cif"), use_seqres=False)
    assert p.chain_id_list == ["A", "B", "C", "D", "E", "F"]

    p = p.renumber_from_gapped_domain(GAPPED_AHO_7K7R)

    for cdr in [f"H{i}" for i in (1, 2, 3, 4)]:
        expected = GAPPED_AHO_7K7R["B"][slice(*CDR_RANGES_AHO[cdr])].replace("-", "")
        selector = CDRResidueSelector(
            which_cdrs=[cdr], heavy_chain="B", light_chain="A", scheme="aho"
        )
        selected = p[selector]
        assert selected.sequence[selector.heavy_chain] == expected

    for cdr in [f"L{i}" for i in (1, 2, 3, 4)]:
        expected = GAPPED_AHO_7K7R["A"][slice(*CDR_RANGES_AHO[cdr])].replace("-", "")
        selector = CDRResidueSelector(
            which_cdrs=[cdr], heavy_chain="B", light_chain="A", scheme="aho"
        )
        selected = p[selector]
        assert selected.sequence[selector.light_chain] == expected
