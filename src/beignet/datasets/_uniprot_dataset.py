from os import PathLike
from pathlib import Path
from typing import Callable

import pooch
from pooch import Decompress

from beignet.transforms import Transform

from ._fasta_dataset import FASTADataset


class UniProtDataset(FASTADataset):
    def __init__(
        self,
        url: str,
        root: str | PathLike | None = None,
        known_hash: str | None = None,
        *,
        transform: Callable | Transform | None = None,
        target_transform: Callable | Transform | None = None,
    ):
        """
        Parameters
        ----------
        url : str
            URL to the file that needs to be downloaded. Ideally, the URL
            should end with a file name (e.g., `uniref50.fasta.gz`).

        root : str | PathLike, optional
            Root directory where the dataset subdirectory exists or, if
            `download` is `True`, the directory where the dataset subdirectory
            will be created and the dataset downloaded.

        transform : Callable | Transform, optional
            A `Callable` or `Transform` that that maps a sequence to a
            transformed sequence (default: `None`).

        target_transform : Callable | Transform, optional
            A `Callable` or `Transform` that maps a target (a cluster
            identifier) to a transformed target (default: `None`).
        """
        if root is None:
            root = pooch.os_cache("beignet")

        if isinstance(root, str):
            root = Path(root)

        self.root = root.resolve()

        name = self.__class__.__name__.replace("Dataset", "")

        super().__init__(
            pooch.retrieve(
                url,
                known_hash,
                f"{name}.fasta.gz",
                root / name,
                processor=Decompress(
                    name=f"{name}.fasta",
                ),
                progressbar=True,
            ),
        )

        self.transform = transform

        self.target_transform = target_transform

    def __getitem__(self, index: int) -> (str, str):
        input, target = self.get(index)

        if self.transform:
            input = self.transform(input)

        if self.target_transform:
            target = self.target_transform(target)

        return input, target
