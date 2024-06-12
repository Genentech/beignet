import functools
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from ..__safe_sum import _safe_sum
from ..__zero_diagonal_mask import _zero_diagonal_mask
from .__kwargs_to_pair_parameters import (
    _kwargs_to_pair_parameters,
)
from .__map_product import _map_product
from .__merge_dictionaries import _merge_dictionaries


def _pair_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    kinds: Optional[Union[int, Tensor]] = None,
    dim: Optional[Tuple[int, ...]] = None,
    keepdim: bool = False,
    ignore_unused_parameters: bool = False,
    **kwargs,
) -> Callable[..., Tensor]:
    parameters, combinators = {}, {}

    for name, parameter in kwargs.items():
        if isinstance(parameter, Callable):
            combinators[name] = parameter
        elif isinstance(parameter, tuple) and isinstance(parameter[0], Callable):
            assert len(parameter) == 2

            combinators[name], parameters[name] = parameter[0], parameter[1]
        else:
            parameters[name] = parameter

    merge_dicts = functools.partial(
        _merge_dictionaries,
        ignore_unused_parameters=ignore_unused_parameters,
    )

    if kinds is None:

        def mapped_fn(_position: Tensor, **_dynamic_kwargs) -> Tensor:
            distance_fn = functools.partial(displacement_fn, **_dynamic_kwargs)

            distances = _map_product(distance_fn)(_position, _position)

            dictionaries = merge_dicts(parameters, _dynamic_kwargs)

            to_parameters = _kwargs_to_pair_parameters(
                dictionaries,
                combinators,
            )

            u = fn(distances, **to_parameters)

            u = _zero_diagonal_mask(u)

            u = _safe_sum(u, dim=dim, keepdim=keepdim)

            return u * 0.5

        return mapped_fn

    if isinstance(kinds, Tensor):
        if not isinstance(kinds, Tensor) or kinds.is_floating_point():
            raise ValueError

        kinds_count = int(torch.max(kinds))

        if dim is not None or keepdim:
            raise ValueError

        def mapped_fn(_position: Tensor, **_dynamic_kwargs):
            u = torch.tensor(0.0, dtype=torch.float32)

            distance_fn = functools.partial(displacement_fn, **_dynamic_kwargs)

            distance_fn = _map_product(distance_fn)

            for m in range(kinds_count + 1):
                for n in range(m, kinds_count + 1):
                    distance = distance_fn(
                        _position[kinds == m],
                        _position[kinds == n],
                    )

                    _kwargs = merge_dicts(parameters, _dynamic_kwargs)

                    s_kwargs = _kwargs_to_pair_parameters(_kwargs, combinators, (m, n))

                    u = fn(distance, **s_kwargs)

                    if m == n:
                        u = _zero_diagonal_mask(u)

                        u = _safe_sum(u)

                        u = u + u * 0.5
                    else:
                        y = _safe_sum(u)

                        u = u + y

            return u

        return mapped_fn

    if isinstance(kinds, int):
        kinds_count = kinds

        def mapped_fn(_position: Tensor, _kinds: Tensor, **_dynamic_kwargs):
            if not isinstance(_kinds, Tensor) or _kinds.is_floating_point():
                raise ValueError

            u = torch.tensor(0.0, dtype=torch.float32)

            n = _position.shape[0]

            distance_fn = functools.partial(displacement_fn, **_dynamic_kwargs)

            distance_fn = _map_product(distance_fn)

            _kwargs = merge_dicts(parameters, _dynamic_kwargs)

            distance = distance_fn(_position, _position)

            for m in range(kinds_count):
                for n in range(kinds_count):
                    a = torch.reshape(
                        _kinds == m,
                        [
                            n,
                        ],
                    )
                    b = torch.reshape(
                        _kinds == n,
                        [
                            n,
                        ],
                    )

                    a = a.to(dtype=_position.dtype)[:, None]
                    b = b.to(dtype=_position.dtype)[None, :]

                    mask = a * b

                    if m == n:
                        mask = _zero_diagonal_mask(mask) * mask

                    to_parameters = _kwargs_to_pair_parameters(
                        _kwargs, combinators, (m, n)
                    )

                    y = fn(distance, **to_parameters) * mask

                    y = _safe_sum(y, dim=dim, keepdim=keepdim)

                    u = u + y

            return u / 2.0

        return mapped_fn

    raise ValueError
