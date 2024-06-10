from os import PathLike

from numpy.typing import ArrayLike

from ._sequence_dataset import SequenceDataset


class SizedSequenceDataset(SequenceDataset):
    def __init__(
        self,
        root: str | PathLike,
        sizes: ArrayLike,
        *args,
        **kwargs,
    ):
        super().__init__(root, *args, **kwargs)

        self.sizes = sizes

    def __len__(self) -> int:
        return len(self.sizes)
