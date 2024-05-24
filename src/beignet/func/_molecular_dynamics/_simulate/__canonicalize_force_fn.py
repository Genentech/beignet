from typing import Callable

import optree
from torch import Tensor

from .._force import force


def _canonicalize_force_fn(fn: Callable[..., Tensor]) -> Callable[..., Tensor]:
    _force_fn = None

    def _fn(_positions: Tensor, **kwargs):
        nonlocal _force_fn

        if _force_fn is None:
            outputs = fn(_positions, **kwargs)

            if outputs.shape == ():
                _force_fn = force(fn)
            else:

                def _f(x: Tensor, y: Tensor) -> bool:
                    return x.shape == y.shape

                tree_map = optree.tree_map(_f, outputs, _positions)

                def _g(x, y):
                    return x and y

                if not optree.tree_reduce(_g, tree_map, True):
                    raise ValueError

                _force_fn = fn

        return _force_fn(_positions, **kwargs)

    return _fn
