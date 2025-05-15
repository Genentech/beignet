import json
import pathlib
from pprint import pprint

import pytest
import torch

from beignet.structure import (
    ResidueArray,
    dockq_contact_score,
    dockq_irmsd_score,
    dockq_lrmsd_score,
)
from beignet.structure.selectors import ChainSelector


@pytest.fixture
def dockq_test_data_path():
    return pathlib.Path(__file__).parent / "data"


def test_dockq_contact_score(dockq_test_data_path):
    native = ResidueArray.from_pdb(dockq_test_data_path / "1A2K_r_l_b.pdb")
    native = torch.cat(
        [
            native[ChainSelector(["A"])],
            native[ChainSelector(["B"])],
            native[ChainSelector(["C"])],
        ]
    )

    # reorder chains to match
    model = ResidueArray.from_pdb(dockq_test_data_path / "1A2K_r_l_b.model.pdb")
    model = torch.cat(
        [
            model[ChainSelector(["B"])].rename_chains({"B": "A"}),
            model[ChainSelector(["A"])].rename_chains({"A": "B"}),
            model[ChainSelector(["C"])],
        ]
    )

    with open(dockq_test_data_path / "dockq_ref.json", "r") as f:
        ref = json.load(f)

    pprint(ref["best_result"]["BC"])

    results = dockq_contact_score(
        model, native, receptor_chains=["B"], ligand_chains=["C"]
    )

    pprint(results)

    assert results["model_contacts"].item() == ref["best_result"]["BC"]["model_total"]
    assert results["native_contacts"].item() == ref["best_result"]["BC"]["nat_total"]
    assert results["shared_contacts"].item() == ref["best_result"]["BC"]["nat_correct"]
    assert (
        results["non_native_contacts"].item()
        == ref["best_result"]["BC"]["nonnat_count"]
    )


def test_dockq_irmsd_score(dockq_test_data_path):
    native = ResidueArray.from_pdb(
        dockq_test_data_path / "1A2K_r_l_b.pdb", dtype=torch.float64
    )
    native = torch.cat(
        [
            native[ChainSelector(["A"])],
            native[ChainSelector(["B"])],
            native[ChainSelector(["C"])],
        ]
    )

    # reorder chains to match
    model = ResidueArray.from_pdb(
        dockq_test_data_path / "1A2K_r_l_b.model.pdb", dtype=torch.float64
    )
    model = torch.cat(
        [
            model[ChainSelector(["B"])].rename_chains({"B": "A"}),
            model[ChainSelector(["A"])].rename_chains({"A": "B"}),
            model[ChainSelector(["C"])],
        ]
    )

    with open(dockq_test_data_path / "dockq_ref.json", "r") as f:
        ref = json.load(f)

    pprint(ref["best_result"]["BC"])

    irmsd = dockq_irmsd_score(
        model, native, receptor_chains=["B"], ligand_chains=["C"]
    )["interface_rmsd"]

    pprint(f"{irmsd.item()=:0.2e}")

    assert irmsd.item() == pytest.approx(ref["best_result"]["BC"]["iRMSD"], abs=1e-5)


def test_dockq_lrmsd_score(dockq_test_data_path):
    native = ResidueArray.from_pdb(
        dockq_test_data_path / "1A2K_r_l_b.pdb", dtype=torch.float64
    )
    native = torch.cat(
        [
            native[ChainSelector(["A"])],
            native[ChainSelector(["B"])],
            native[ChainSelector(["C"])],
        ]
    )

    # reorder chains to match
    model = ResidueArray.from_pdb(
        dockq_test_data_path / "1A2K_r_l_b.model.pdb", dtype=torch.float64
    )
    model = torch.cat(
        [
            model[ChainSelector(["B"])].rename_chains({"B": "A"}),
            model[ChainSelector(["A"])].rename_chains({"A": "B"}),
            model[ChainSelector(["C"])],
        ]
    )

    with open(dockq_test_data_path / "dockq_ref.json", "r") as f:
        ref = json.load(f)

    pprint(ref["best_result"]["BC"])

    lrmsd = dockq_lrmsd_score(
        model, native, receptor_chains=["C"], ligand_chains=["B"]
    )["ligand_rmsd"]

    pprint(f"{lrmsd.item()=:0.2e}")

    assert lrmsd.item() == pytest.approx(ref["best_result"]["BC"]["LRMSD"], abs=1e-3)
