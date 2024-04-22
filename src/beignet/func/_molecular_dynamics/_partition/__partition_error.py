import torch
from torch import Tensor

from ..__dataclass import _dataclass
from .__partition_error_kind import _PartitionErrorKind


@_dataclass
class _PartitionError:
    code: Tensor

    def update(self, bit: bytes, predicate: Tensor) -> "_PartitionError":
        zero = torch.zeros([], dtype=torch.uint8)

        bit = torch.tensor(bit, dtype=torch.uint8)

        return _PartitionError(code=self.code | torch.where(predicate, bit, zero))

    def __str__(self) -> str:
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
