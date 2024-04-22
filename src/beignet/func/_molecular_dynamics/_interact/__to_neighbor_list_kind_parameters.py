import functools

import optree
import torch
from optree import PyTree
from torch import Tensor

from .._partition.__is_neighbor_list_sparse import _is_neighbor_list_sparse
from .._partition.__map_bond import _map_bond
from .._partition.__neighbor_list_format import _NeighborListFormat
from .__parameter_tree import _ParameterTree
from .__parameter_tree_kind import _ParameterTreeKind


def _to_neighbor_list_kind_parameters(
    format: _NeighborListFormat,
    indexes: Tensor,
    kinds: Tensor,
    parameters: _ParameterTree | Tensor | float,
) -> PyTree | _ParameterTree | Tensor | float:
    fn = functools.partial(
        lambda p, a, b: p[a, b],
        parameters,
    )

    match parameters:
        case parameters if isinstance(parameters, Tensor):
            match parameters.shape:
                case 0:
                    return parameters
                case 2:
                    if _is_neighbor_list_sparse(format):
                        return _map_bond(
                            fn,
                        )(
                            kinds[indexes[0]],
                            kinds[indexes[1]],
                        )

                    return torch.func.vmap(
                        torch.func.vmap(
                            fn,
                            in_dims=(None, 0),
                        ),
                    )(kinds, kinds[indexes])
                case _:
                    raise ValueError
        case parameters if isinstance(parameters, _ParameterTree):
            match parameters.kind:
                case _ParameterTreeKind.KINDS:
                    if _is_neighbor_list_sparse(format):
                        return optree.tree_map(
                            lambda parameter: _map_bond(
                                functools.partial(
                                    fn,
                                    parameter,
                                ),
                            )(
                                kinds[indexes[0]],
                                kinds[indexes[1]],
                            ),
                            parameters.tree,
                        )

                    return optree.tree_map(
                        lambda parameter: torch.func.vmap(
                            torch.func.vmap(
                                functools.partial(
                                    fn,
                                    parameter,
                                ),
                                (None, 0),
                            )
                        )(
                            kinds,
                            kinds[indexes],
                        ),
                        parameters.tree,
                    )
                case _ParameterTreeKind.SPACE:
                    return parameters.tree
        case _:
            raise ValueError

    return parameters
