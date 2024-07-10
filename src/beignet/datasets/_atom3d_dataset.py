from pathlib import Path
from typing import Callable, Optional, Union

import beignet.io
from beignet.transforms import Transform

from ._lmdb_dataset import LMDBDataset


class ATOM3DDataset(LMDBDataset):
    def __init__(
        self,
        root: Union[str, Path],
        path: Union[str, Path],
        resource: str,
        name: str,
        *,
        checksum: Optional[str] = None,
        download: bool = False,
        transform: Union[Callable, Transform, None] = None,
    ):
        self._root = root

        if isinstance(self._root, str):
            self._root = Path(self._root).resolve()

        self._transform_fn = transform

        if download:
            beignet.io.download_and_extract_archive(
                resource,
                self._root / f"ATOM3D{name}",
                checksum=checksum,
            )

        super().__init__(
            self._root / f"ATOM3D{name}" / path,
            transform=transform,
        )
