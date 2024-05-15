import math
import random
from collections import defaultdict

from torch.utils.data import Dataset, Subset

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    _RDKit_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _RDKit_AVAILABLE = False
    Chem, MurckoScaffoldSmiles = None, None


def murcko_scaffold_split(
    dataset: Dataset,
    smiles: list[str],
    test_size: float | int,
    *,
    seed: int = 0xDEADBEEF,
    shuffle: bool = True,
    include_chirality: bool = False,
) -> tuple[Subset, Subset]:
    """Split a dataset based on Murcko scaffold splitting based
    on provided SMILES strings.

    Note that for datasets that are small or not highly diverse,
    the final test set may be smaller than the specified test_size.

    Parameters:
        dataset (Dataset): The dataset to split.
        smiles (list[str]): A list of SMILES strings.
        test_size (float | int): The size of the test set.
            If float, should be between 0.0 and 1.0.
            If int, should be between 0 and len(smiles).
        seed (int): The random seed to use for shuffling.
        shuffle (bool): Whether to shuffle the indices.
        include_chirality (bool): Whether to include chirality in the scaffold.

    Returns:
        tuple[Subset, Subset]: The train and test subsets.

    """
    train_idx, test_idx = _murcko_scaffold_split_indices(
        smiles,
        test_size,
        seed=seed,
        shuffle=shuffle,
        include_chirality=include_chirality,
    )
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def _murcko_scaffold_split_indices(
    smiles: list[str],
    test_size: float | int,
    *,
    seed: int = 0xDEADBEEF,
    shuffle: bool = True,
    include_chirality: bool = False,
) -> tuple[list[int], list[int]]:
    """Get train and test indices based on Murcko scaffold splitting."""
    if not _RDKit_AVAILABLE:
        raise ImportError(
            "This function requires RDKit to be installed (pip install rdkit)"
        )

    if (
        isinstance(test_size, int) and (test_size <= 0 or test_size >= len(smiles))
    ) or (isinstance(test_size, float) and (test_size <= 0 or test_size >= 1)):
        raise ValueError(
            f"Test_size should be a float in (0, 1) or and int < {len(smiles)}."
        )

    if isinstance(test_size, float):
        test_size = math.ceil(len(smiles) * test_size)

    scaffolds = defaultdict(list)

    for ind, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        scaffolds[scaffold].append(ind)

    train_idx = []
    test_idx = []

    if shuffle:
        if seed is not None:
            random.Random(seed).shuffle(scaffolds)
        else:
            random.shuffle(scaffolds)

    for index_list in scaffolds.values():
        if len(test_idx) + len(index_list) <= test_size:
            test_idx = [*test_idx, *index_list]
        else:
            train_idx.extend(index_list)

    return train_idx, test_idx
