import os.path
import re
from pathlib import Path
from typing import Callable, Optional, Union

from beignet.io import download_and_extract_archive

from ._fasta_dataset import FASTADataset


class UniRefDataset(FASTADataset):
    def __init__(
        self,
        root: Union[str, Path],
        name: str,
        md5: tuple[str, str],
        *,
        index: bool = True,
        download: bool = False,
        transform_fn: Optional[Callable] = None,
        target_transform_fn: Optional[Callable] = None,
    ) -> None:
        """
        :param root: Root directory where the dataset subdirectory exists or,
            if :attr:`download` is ``True``, the directory where the dataset
            subdirectory will be created and the dataset downloaded.

        :param name:

        :param md5:

        :param index: If ``True``, caches the sequence
            indicies to disk for faster re-initialization (default: ``True``).

        :param download: If ``True``, download the dataset and to the
            :attr:`root` directory (default: ``False``). If the dataset is
            already downloaded, it is not redownloaded.

        :param transform_fn: A ``Callable`` that maps a sequence to a
            transformed sequence (default: ``None``).

        :param target_transform_fn: ``Callable`` that maps a target (a cluster
            identifier) to a transformed target (default: ``None``).
        """
        root = Path(root)

        directory = root / name

        path = directory / f"{name}.fasta"

        if download and not os.path.exists(path):
            download_and_extract_archive(
                f"http://ftp.uniprot.org/pub/databases/uniprot/uniref/{name}/{name}.fasta.gz",
                str(directory),
                str(directory),
                f"{name}.fasta.gz",
                md5[1],
            )

        self._pattern = re.compile(r"^UniRef.+_([A-Z0-9]+)\s.+$")

        super().__init__(
            path,
            index=index,
        )

        self._transform_fn = transform_fn

        self._target_transform_fn = target_transform_fn

    def __getitem__(self, index: int) -> tuple[str, str]:
        target, sequence = self.get(index)

        (target,) = re.search(self._pattern, target).groups()

        if self._transform_fn:
            sequence = self._transform_fn(sequence)

        if self._target_transform_fn:
            target = self._target_transform_fn(target)

        return sequence, target
