from typing import Dict

import optree
from torch import Tensor

from .__parameter_tree import _ParameterTree
from .__parameter_tree_kind import _ParameterTreeKind


def _to_bond_kind_parameters(
    parameter: Tensor | _ParameterTree,
    kinds: Tensor,
) -> Tensor | _ParameterTree:
    assert isinstance(kinds, Tensor)

    assert len(kinds.shape) == 1

    match parameter:
        case Tensor():
            match parameter.shape:
                case 0:
                    return parameter
                case 1:
                    return parameter[kinds]
                case _:
                    raise ValueError
        case _ParameterTree():
            if parameter.kind is _ParameterTreeKind.BOND:

                def _fn(_parameter: Dict) -> Tensor:
                    return _parameter[kinds]

                return optree.tree_map(_fn, parameter.tree)

            if parameter.kind is _ParameterTreeKind.SPACE:
                return parameter.tree

            raise ValueError
        case float() | int():
            return parameter
        case _:
            raise NotImplementedError
