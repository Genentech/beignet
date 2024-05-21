from os import PathLike
from pathlib import Path
from typing import Callable

import pooch
from pooch import Decompress

from beignet.datasets import FASTADataset
from beignet.transforms import Transform


class CaLMDataset(FASTADataset):
    def __init__(
        self,
        root: str | PathLike | None = None,
        *,
        train: bool = True,
        transform: Callable | Transform | None = None,
        target_transform: Callable | Transform | None = None,
    ):
        """
        Parameters
        ----------
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

        self._root = root.resolve()

        name = self.__class__.__name__.replace("Dataset", "")

        if train:
            path = pooch.retrieve(
                "http://opig.stats.ox.ac.uk/data/downloads/training_data.tar.gz",
                "",
                f"{name}-train.fasta.gz",
                root / name,
                processor=Decompress(
                    name=f"{name}-train.fasta",
                ),
                progressbar=True,
            )
        else:
            path = pooch.retrieve(
                "http://opig.stats.ox.ac.uk/data/downloads/heldout.tar.gz",
                "",
                f"{name}-test.fasta.gz",
                root / name,
                processor=Decompress(
                    name=f"{name}-test.fasta",
                ),
                progressbar=True,
            )

        super().__init__(path)

        self.transform = transform

        # self.target_pattern = re.compile(r"^Calm.+_([A-Z0-9]+)\s.+$")

        self.target_transform = target_transform
