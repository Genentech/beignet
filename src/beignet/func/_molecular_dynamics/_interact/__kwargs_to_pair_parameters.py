from typing import Callable, Dict, Union

import optree
from optree import PyTree
from torch import Tensor

from .__parameter_tree import _ParameterTree
from .__parameter_tree_kind import _ParameterTreeKind


def _kwargs_to_pair_parameters(
    kwargs: Dict[str, Union[_ParameterTree, Tensor, float, PyTree]],
    combinators: Dict[str, Callable],
    kinds: Tensor | None = None,
) -> Dict[str, Tensor]:
    parameters = {}

    for name, parameter in kwargs.items():
        if kinds is None:

            def _combinator_fn(x: Tensor, y: Tensor) -> Tensor:
                return (x + y) * 0.5

            combinator = combinators.get(name, _combinator_fn)

            match parameter:
                case _ParameterTree():
                    match parameter.kind:
                        case _ParameterTreeKind.BOND | _ParameterTreeKind.SPACE:
                            parameters[name] = parameter.tree
                        case _ParameterTreeKind.PARTICLE:

                            def _particle_fn(_parameter: Tensor) -> Tensor:
                                return combinator(
                                    _parameter[:, None, ...],
                                    _parameter[None, :, ...],
                                )

                            parameters[name] = optree.tree_map(
                                _particle_fn,
                                parameter.tree,
                            )
                        case _:
                            message = f"""
parameter `kind` is `{parameter.kind}`. If `kinds` is `None` and a parameter is
an instance of `ParameterTree`, `kind` must be `ParameterTreeKind.BOND`,
`ParameterTreeKind.PARTICLE`, or `ParameterTreeKind.SPACE`.
                            """.replace("\n", " ")

                            raise ValueError(message)
                case Tensor():
                    match parameter.ndim:
                        case 0 | 2:
                            parameters[name] = parameter
                        case 1:
                            parameters[name] = combinator(
                                parameter[:, None],
                                parameter[None, :],
                            )
                        case _:
                            message = f"""
parameter `ndim` is `{parameter.ndim}`. If `kinds` is `None` and a parameter is
an instance of `Tensor`, `ndim` must be in `0`, `1`, or `2`.
                            """.replace("\n", " ")

                            raise ValueError(message)
                case float() | int():
                    parameters[name] = parameter
                case _:
                    message = f"""
parameter `type` is {type(parameter)}. If `kinds` is `None`, a parameter must
be an instance of `ParameterTree`, `Tensor`, `float`, or `int`.
                    """.replace("\n", " ")

                    raise ValueError(message)
        else:
            if name in combinators:
                raise ValueError

            match parameter:
                case _ParameterTree():
                    match parameter.kind:
                        case _ParameterTreeKind.SPACE:
                            parameters[name] = parameter.tree
                        case _ParameterTreeKind.KINDS:

                            def _kinds_fn(_parameter: Tensor) -> Tensor:
                                return _parameter[kinds]

                            parameters[name] = optree.tree_map(
                                _kinds_fn,
                                parameter.tree,
                            )
                        case _:
                            message = f"""
parameter `kind` is {parameter.kind}. If `kinds` is `None` and a parameter is
an instance of `ParameterTree`, `kind` must be `ParameterTreeKind.SPACE` or
`ParameterTreeKind.KINDS`.
                            """.replace("\n", " ")

                            raise ValueError(message)
                case Tensor():
                    match parameter.ndim:
                        case 0:
                            parameters[name] = parameter
                        case 2:
                            parameters[name] = parameter[kinds]
                        case _:
                            message = f"""
parameter `ndim` is `{parameter.ndim}`. If `kinds` is not `None` and a
parameter is an instance of `Tensor`, `ndim` must be in `0`, `1`, or `2`.
                            """.replace("\n", " ")

                            raise ValueError(message)
                case _:
                    parameters[name] = parameter

    return parameters
