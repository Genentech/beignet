import torch
from torch import Tensor

from ..__dataclass import _dataclass
from .__partition_error_kind import _PartitionErrorKind


@_dataclass
class _PartitionError:
    r"""A class to represent and manage partition errors with specific error codes.

    Attributes:
    -----------
    code : Tensor
        A tensor representing the error code.

    Methods:
    --------
    update(bit: bytes, predicate: Tensor) -> "_PartitionError"
        Update the error code based on a predicate and a new bit.

    __str__() -> str
        Provide a human-readable string representation of the error.

    __repr__() -> str
        Alias for __str__().
    """

    code: Tensor

    def update(self, bit: bytes, predicate: Tensor) -> "_PartitionError":
        r"""Update the error code based on a predicate and a new bit.

        Parameters:
        -----------
        bit : bytes
            The bit to be combined with the existing error code.
        predicate : Tensor
            A tensor that determines where the bit should be applied.

        Returns:
        --------
        _PartitionError
            A new instance of `_PartitionError` with the updated error code.
        """
        zero = torch.zeros([], dtype=torch.uint8)

        bit = torch.tensor(bit, dtype=torch.uint8)

        return _PartitionError(code=self.code | torch.where(predicate, bit, zero))

    def __str__(self) -> str:
        r"""Provide a human-readable string representation of the error.

        Returns:
        --------
        str
            A string describing the error.

        Raises:
        -------
        ValueError
            If the error code is unexpected or not recognized.
        """
        if not torch.any(self.code):
            return ""

        if torch.any(self.code & _PartitionErrorKind.NEIGHBOR_LIST_OVERFLOW):
            return "Partition Error: Neighbor list buffer overflow."

        if torch.any(self.code & _PartitionErrorKind.CELL_LIST_OVERFLOW):
            return "Partition Error: Cell list buffer overflow"

        if torch.any(self.code & _PartitionErrorKind.CELL_SIZE_TOO_SMALL):
            return "Partition Error: Cell size too small"

        if torch.any(self.code & _PartitionErrorKind.MALFORMED_BOX):
            return "Partition Error: Incorrect box format. Expecting upper triangular."

        raise ValueError(f"Unexpected Error Code {self.code}.")

    __repr__ = __str__
