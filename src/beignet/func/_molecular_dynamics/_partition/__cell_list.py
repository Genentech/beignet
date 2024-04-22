from typing import Callable, Dict

from torch import Tensor

from ..__dataclass import _dataclass
from .._static_field import static_field


@_dataclass
class _CellList:
    exceeded_maximum_size: Tensor

    indexes: Tensor

    item_size: float = static_field()

    parameters: Dict[str, Tensor]

    positions_buffer: Tensor

    size: int = static_field()

    update_fn: Callable[..., "_CellList"] = static_field()

    def update(self, positions: Tensor, **kwargs) -> "_CellList":
        return self.update_fn(
            positions,
            [
                self.size,
                self.exceeded_maximum_size,
                self.update_fn,
            ],
            **kwargs,
        )
