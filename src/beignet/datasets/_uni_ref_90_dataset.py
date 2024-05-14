from pathlib import Path
from typing import Callable, Optional, Union

from ._uni_ref_dataset import UniRefDataset


class UniRef90Dataset(UniRefDataset):
    def __init__(
        self,
        root: Union[str, Path],
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
        super().__init__(
            root,
            "uniref90",
            (
                "",
                "",
            ),
            index=index,
            download=download,
            transform_fn=transform_fn,
            target_transform_fn=target_transform_fn,
        )
