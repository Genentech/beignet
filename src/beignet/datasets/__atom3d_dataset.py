from os import PathLike
from pathlib import Path
from typing import Callable

import pooch

from beignet.transforms import Transform

from ._lmdb_dataset import LMDBDataset


class ATOM3DDataset(LMDBDataset):
    def __init__(
        self,
        root: str | PathLike,
        path: str | PathLike,
        resource: str,
        name: str,
        *,
        checksum: str | None = None,
        download: bool = False,
        transform: Callable | Transform | None = None,
    ):
        if root is None:
            root = pooch.os_cache("beignet")

        if isinstance(root, str):
            root = Path(root)

        self.root = root.resolve()

        self.transform = transform

        if download:
            pooch.retrieve(
                resource,
                self.root / f"ATOM3D{name}",
                checksum=checksum,
            )

        super().__init__(
            self.root / f"ATOM3D{name}" / path,
            transform=transform,
        )
