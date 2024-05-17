from importlib.util import find_spec
from unittest.mock import MagicMock, patch

import pytest
from beignet.subsets._murcko_scaffold_split import (
    _murcko_scaffold_split_indices,
    murcko_scaffold_split,
)
from torch.utils.data import Dataset, Subset

_RDKit_AVAILABLE = find_spec("rdkit") is not None


@pytest.mark.skipif(not _RDKit_AVAILABLE, reason="RDKit is not available")
@patch("beignet.subsets._murcko_scaffold_split._murcko_scaffold_split_indices")
def test_murcko_scaffold_split(mock__murcko_scaffold_split_indices):
    mock__murcko_scaffold_split_indices.return_value = ([0], [1])

    mock_dataset = MagicMock(spec=Dataset)

    train_dataset, test_dataset = murcko_scaffold_split(
        dataset=mock_dataset,
        smiles=["C", "C"],
        test_size=0.5,
        shuffle=False,
        seed=0,
    )

    assert isinstance(train_dataset, Subset)
    assert isinstance(test_dataset, Subset)
    assert train_dataset.indices == [0]
    assert test_dataset.indices == [1]


@pytest.mark.skipif(not _RDKit_AVAILABLE, reason="RDKit is not available")
@pytest.mark.parametrize(
    "test_size, expected_train_idx, expected_test_idx",
    [
        pytest.param(0.5, [2, 3], [0, 1], id="test_size is float"),
        pytest.param(2, [2, 3], [0, 1], id="test_size is int"),
    ],
)
def test__murcko_scaffold_split_indices(
    test_size, expected_train_idx, expected_test_idx
):
    smiles = ["C1CCCCC1", "C1CCCCC1", "CCO", "CCO"]

    train_idx, test_idx = _murcko_scaffold_split_indices(
        smiles,
        test_size=test_size,
    )
    assert train_idx == expected_train_idx
    assert test_idx == expected_test_idx


@pytest.mark.skipif(not _RDKit_AVAILABLE, reason="RDKit is not available")
@pytest.mark.parametrize(
    "smiles, test_size",
    [
        pytest.param(["CCO"], 1.2, id="test_size is float > 1"),
        pytest.param(["CCO"], -1, id="test_size is negative"),
        pytest.param(["CCO"], 0, id="test_size is 0"),
        pytest.param(["CCO"], 5, id="test_size > len(smiles)"),
    ],
)
def test__murcko_scaffold_split_indices_invalid_inputs(smiles, test_size):
    with pytest.raises(ValueError):
        _murcko_scaffold_split_indices(
            smiles,
            test_size=test_size,
        )
