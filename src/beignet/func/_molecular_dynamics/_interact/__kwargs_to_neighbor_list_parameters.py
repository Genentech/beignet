from typing import Callable, Dict

from torch import Tensor

from .._partition.__neighbor_list_format import _NeighborListFormat
from .__to_neighbor_list_kind_parameters import (
    _to_neighbor_list_kind_parameters,
)
from .__to_neighbor_list_matrix_parameters import (
    _to_neighbor_list_matrix_parameters,
)


def _kwargs_to_neighbor_list_parameters(
    format: _NeighborListFormat,
    indexes: Tensor,
    species: Tensor,
    kwargs: Dict[str, Tensor],
    combinators: Dict[str, Callable],
) -> Dict[str, Tensor]:
    parameters = {}

    for name, parameter in kwargs.items():
        if species is None or (isinstance(parameter, Tensor) and parameter.ndim == 1):
            combinator = combinators.get(name, lambda x, y: 0.5 * (x + y))

            parameters[name] = _to_neighbor_list_matrix_parameters(
                format,
                indexes,
                parameter,
                combinator,
            )
        else:
            if name in combinators:
                raise ValueError

            parameters[name] = _to_neighbor_list_kind_parameters(
                format,
                indexes,
                species,
                parameter,
            )

    return parameters
