from pathlib import Path

import numpy

from ._sequence_dataset import SequenceDataset


class SizedSequenceDataset(SequenceDataset):
    def __init__(
        self,
        root: str | Path,
        sizes: numpy.ndarray,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(root, *args, **kwargs)

        self._sizes = sizes

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.sizes)

    @property
    def sizes(self) -> numpy.ndarray:
        return self._sizes
