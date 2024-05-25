import shutil
from os import PathLike
from pathlib import Path
from typing import Callable

import pooch
from pooch import Decompress, Untar

from beignet.transforms import Transform

from ._fasta_dataset import FASTADataset


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
                "https://opig.stats.ox.ac.uk/data/downloads/training_data.tar.gz",
                "sha256:22673dc6db6fa0dfa9fb6d2b1e94fe2a94f01e0a726e597f03d9b31d1b503f0e",
                f"{name}-train.fasta.gz",
                root / name,
                processor=Decompress(
                    name=f"{name}-train.fasta",
                ),
                progressbar=True,
            )
        else:
            pooch.retrieve(
                "https://opig.stats.ox.ac.uk/data/downloads/heldout.tar.gz",
                "sha256:6433a04c75aa16b555bb5fe2e0501315e5e98811d19447f6f8bc05939e8cb23d",
                f"{name}-test.fasta.gz",
                root / name,
                processor=Untar(
                    extract_dir=f"{name}-test",
                ),
                progressbar=True,
            )

            shutil.move(
                f"{root}/{name}/{name}-test/hsapiens.fasta",
                f"{root}/{name}/{name}-test.fasta",
            )

            path = root / name / f"{name}-test.fasta"

        super().__init__(path)

        self.transform = transform

        self.target_transform = target_transform
