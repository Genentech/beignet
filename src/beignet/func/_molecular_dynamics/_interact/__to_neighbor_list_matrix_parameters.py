import functools
from typing import Callable

import optree
import torch
from optree import PyTree
from torch import Tensor

from .._partition.__is_neighbor_list_sparse import _is_neighbor_list_sparse
from .._partition.__map_bond import _map_bond
from .._partition.__map_neighbor import _map_neighbor
from .._partition.__neighbor_list_format import _NeighborListFormat
from .__parameter_tree import _ParameterTree
from .__parameter_tree_kind import _ParameterTreeKind


def _to_neighbor_list_matrix_parameters(
    format: _NeighborListFormat,
    indexes: Tensor,
    parameters: _ParameterTree | Tensor | float,
    combinator: Callable[[Tensor, Tensor], Tensor],
) -> PyTree | _ParameterTree | Tensor | float:
    match parameters:
        case parameters if isinstance(parameters, Tensor):
            match parameters.ndim:
                case 0:
                    return parameters
                case 1:
                    if _is_neighbor_list_sparse(format):
                        return _map_bond(
                            combinator,
                        )(
                            parameters[indexes[0]],
                            parameters[indexes[1]],
                        )

                    return combinator(
                        parameters[:, None],
                        parameters[indexes],
                    )
                case 2:
                    if _is_neighbor_list_sparse(format):
                        return _map_bond(
                            lambda a, b: parameters[a, b],
                        )(
                            indexes[0],
                            indexes[1],
                        )

                    return torch.func.vmap(
                        torch.func.vmap(
                            lambda a, b: parameters[a, b],
                            (None, 0),
                        ),
                    )(
                        torch.arange(indexes.shape[0], dtype=torch.int32),
                        indexes,
                    )
                case _:
                    raise ValueError
        case parameters if isinstance(parameters, _ParameterTree):
            match parameters.kind:
                case _ParameterTreeKind.BOND:
                    if _is_neighbor_list_sparse(format):
                        return optree.tree_map(
                            lambda parameter: _map_bond(
                                functools.partial(
                                    lambda p, a, b: p[a, b],
                                    parameter,
                                ),
                            )(
                                indexes[0],
                                indexes[1],
                            ),
                            parameters.tree,
                        )

                    return optree.tree_map(
                        lambda parameter: torch.func.vmap(
                            torch.func.vmap(
                                functools.partial(
                                    lambda p, a, b: p[a, b],
                                    parameter,
                                ),
                                (None, 0),
                            ),
                        )(
                            torch.arange(indexes.shape[0], dtype=torch.int32),
                            indexes,
                        ),
                        parameters.tree,
                    )
                case _ParameterTreeKind.PARTICLE:
                    if _is_neighbor_list_sparse(format):
                        return optree.tree_map(
                            lambda parameter: _map_bond(
                                combinator,
                            )(
                                parameter[indexes[0]],
                                parameter[indexes[1]],
                            ),
                            parameters.tree,
                        )

                    return optree.tree_map(
                        lambda parameter: _map_neighbor(
                            combinator,
                        )(
                            parameter,
                            parameter[indexes],
                        ),
                        parameters.tree,
                    )
                case _ParameterTreeKind.SPACE:
                    return parameters.tree
                case _:
                    raise ValueError
        case parameters if isinstance(parameters, float):
            return parameters
        case parameters if isinstance(parameters, int):
            return parameters
        case _:
            raise ValueError
