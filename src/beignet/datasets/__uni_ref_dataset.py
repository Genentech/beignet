import re
from pathlib import Path
from typing import Callable

import pooch

from beignet.transforms import Transform

from ._fasta_dataset import FASTADataset


class _UniRefDataset(FASTADataset):
    def __init__(
        self,
        root: str | Path,
        name: str,
        md5: (str, str),
        *,
        index: bool = True,
        download: bool = False,
        transform: Callable | Transform | None = None,
        target_transform: Callable | Transform | None = None,
    ) -> None:
        """
        Parameters
        ----------
        root : str | Path
            Root directory where the dataset subdirectory exists or, if
            `download` is `True`, the directory where the dataset subdirectory
            will be created and the dataset downloaded.

        index : bool, optional
            If `True`, caches the sequence indicies to disk for faster
            re-initialization (default: `True`).

        download : bool, optional
            If `True`, download the dataset and to the `root` directory
            (default: `False`). If the dataset is already downloaded, it is
            not redownloaded.

        transform : Callable | Transform, optional
            A `Callable` or `Transform` that that maps a sequence to a
            transformed sequence (default: `None`).

        target_transform : Callable | Transform, optional
            A `Callable` or `Transform` that maps a target (a cluster
            identifier) to a transformed target (default: `None`).
        """
        root = Path(root)

        directory = root / name

        path = directory / f"{name}.fasta"

        if download:
            pooch.retrieve(
                f"http://ftp.uniprot.org/pub/databases/uniprot/uniref/{name}/{name}.fasta.gz",
                md5[1],
                f"{name}.fasta.gz",
                root / name,
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
