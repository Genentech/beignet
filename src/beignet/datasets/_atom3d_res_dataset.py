from pathlib import Path
from typing import Callable, Tuple, Union

from pandas import DataFrame

from beignet.transforms import Transform

from ._atom3d_dataset import ATOM3DDataset


class ATOM3DRESDataset(ATOM3DDataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = False,
        transform: Union[Callable, Transform, None] = None,
        target_transform: Union[Callable, Transform, None] = None,
    ):
        """
        ATOM3D Residue Identity (RES) consists of atomic environments
        extracted from non-redundant structures in the Protein Data Bank.
        This is formulated as a classification task where the identity of
        the amino acid in the center of the environment is predicted based
        on all other atoms.

        Each sample is a pair of features and a target, where features is
        the molecule’s atomic coordinates and target is the environments’s
        atomic coordinates

        Parameters
        ----------
        root : Union[str, Path]
            The root directory of the dataset.

        download : bool, optional
            If True, download the dataset from the specified source,
            by default `False`.

        transform : Union[Callable, Transform, None], optional
            The transformation function to be applied to the features,
            by default `None`.

        target_transform : Union[Callable, Transform, None], optional
            The transformation function to be applied to the targets,
            by default `None`.
        """
        super().__init__(
            root,
            "raw/RES/data",
            "https://zenodo.org/record/5026743/files/RES-raw.tar.gz",
            "RES",
            checksum="3d6b6c61efb890a8baa303280b6589d9",
            download=download,
        )

        self._transform_fn = transform

        self._target_transform_fn = target_transform

    def __getitem__(self, index: int) -> Tuple[DataFrame, DataFrame]:
        """
        Parameters
        ----------
        index : int
            The index of the item to retrieve from the dataset.

        Returns
        -------
        Tuple[DataFrame, DataFrame]
            A tuple containing the features and target of the item.
        """
        item = super().__getitem__(index)

        features = DataFrame(**item["atoms"])

        if self._transform_fn is not None:
            features = self._transform_fn(features)

        target = DataFrame(**item["labels"])

        if self._target_transform_fn is not None:
            target = self._target_transform_fn(target)

        return features, target
