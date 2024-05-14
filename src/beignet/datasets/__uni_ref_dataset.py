import re
from os import PathLike
from pathlib import Path
from typing import Callable

import pooch
from pooch import Decompress

from beignet.transforms import Transform

from ._fasta_dataset import FASTADataset


class _UniRefDataset(FASTADataset):
    def __init__(
        self,
        url: str,
        root: str | PathLike | None = None,
        known_hash: str | None = None,
        *,
        index: bool = True,
        transform: Callable | Transform | None = None,
        target_transform: Callable | Transform | None = None,
    ) -> None:
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

        index : bool, optional
            If `True`, caches the sequence indexes to disk for faster
            re-initialization (default: `True`).

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

        root = root.resolve()

        name = self.__class__.__name__.replace("Dataset", "")

        path = pooch.retrieve(
            url,
            known_hash,
            f"{name}.fasta.gz",
            root / name,
            processor=Decompress(),
            progressbar=True,
        )

        self._pattern = re.compile(r"^UniRef.+_([A-Z0-9]+)\s.+$")

        super().__init__(path, index=index)

        self._transform = transform

        self._target_transform = target_transform

    def __getitem__(self, index: int) -> (str, str):
        target, sequence = self.get(index)

        (target,) = re.search(self._pattern, target).groups()

        if self._transform:
            sequence = self._transform(sequence)

        if self._target_transform:
            target = self._target_transform(target)

        return sequence, target
