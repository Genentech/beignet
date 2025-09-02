from typing import Callable

from torch import Tensor
from torch.utils.data import Dataset

from beignet.transforms import Transform


class RandomMSADataset(Dataset):
    def __init__(
        self,
        data: Tensor,
        *,
        transform: Callable | Transform | None = None,
    ):
        super().__init__()

        self.data = data

        self.transform = transform

    def __getitem__(
        self,
        index: int,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
        """

        Parameters
        ----------
        index

        Returns
        -------
        (Tensor, Tensor, Tensor, Tensor, Tensor)
            A tuple of five tensors:

            -   `msa`
                One-hot encoding of the processed MSA, using the same
                classes as `restype`.

            -   `has_deletion`
                Binary feature indicating if there is a deletion to the
                left of each position in the MSA.

            -   `deletion_value`
                Raw deletion counts (the number of deletions to the
                left of each MSA position) are transformed to [0, 1]
                using: 2/π arctan d/3.
        .
        """
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self) -> int:
        return len(self.data)
