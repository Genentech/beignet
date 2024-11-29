import dataclasses
import functools
from enum import Enum
from typing import Callable
from typing import Dict
from typing import Iterable, Optional, Union
from typing import Literal
from typing import Tuple

import optree
import torch
from optree import PyTree
from torch import Tensor

from beignet.func.__dataclass import _dataclass
from beignet.func._partition import (
    _NeighborListFormat,
    map_product,
    _NeighborList,
    _map_bond,
    _map_neighbor,
    is_neighbor_list_sparse,
    _segment_sum,
    safe_index,
)


class _ParameterTreeKind(Enum):
    BOND = 0
    KINDS = 1
    PARTICLE = 2
    SPACE = 3


@_dataclass
class _ParameterTree:
    tree: PyTree
    kind: _ParameterTreeKind = dataclasses.field(metadata={"static": True})


def _zero_diagonal_mask(input: Tensor) -> Tensor:
    r"""Creates a zero mask over the diagoonal of a square matrix.

    Parameters:
    -----------
    input : Tensor
        Square matrix.

    Returns:
    --------
    Tensor
        The input matrix with a masked diagonal of zeros.
    """
    if input.shape[0] != input.shape[1]:
        raise ValueError(
            f"Diagonal mask can only mask square matrices. Found {input.shape[0]}x{input.shape[1]}."
        )

    if len(input.shape) > 3:
        raise ValueError(
            f"Diagonal mask can only mask rank-2 or rank-3 tensors. Found {len(input.shape)}."
        )

    n = input.shape[0]

    input = torch.nan_to_num(input)

    mask = 1.0 - torch.eye(n, device=input.device, dtype=input.dtype)

    if len(input.shape) == 3:
        mask = torch.reshape(mask, [n, n, 1])

    return input * mask


def _safe_sum(
    input: Tensor,
    dimension: Optional[Union[Iterable[int], int]] = None,
    keep_dimension: bool = False,
):
    r"""Safely computes the sum of elements in a tensor along a specified
    dimension, promoting the data type to avoid precision loss.

    Parameters
    ----------
    input : Tensor
        The input tensor to be summed.

    dimension : Optional[Union[Iterable[int], int]], optional
        The dimension or dimensions along which to sum. If `None`, sums all elements.
        Default is `None`.

    keep_dimension : bool, optional
        Whether to retain the reduced dimensions in the output tensor. Default is `False`.

    Returns
    -------
    Tensor
        The summed tensor with the same dtype as the input tensor.
    """
    match input:
        case _ if input.is_complex():
            promoted_dtype = torch.complex128
        case _ if input.is_floating_point():
            promoted_dtype = torch.float64
        case _:
            promoted_dtype = torch.int64

    if dimension == ():
        return input.to(dtype=promoted_dtype)

    summation = torch.sum(
        input, dim=dimension, dtype=promoted_dtype, keepdim=keep_dimension
    )

    return summation.to(dtype=input.dtype)


def _force(energy_fn: Callable) -> Callable:
    r"""Computes the force as the negative gradient of an energy."""

    def compute_force(R, *args, **kwargs):
        R = R.requires_grad_(True)
        energy = energy_fn(R, *args, **kwargs)
        force = -torch.autograd.grad(
            energy, R, grad_outputs=torch.ones_like(energy), create_graph=True
        )[0]

        return force

    return compute_force


def _bond_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    static_bonds: Optional[Tensor] = None,
    static_kinds: Optional[Tensor] = None,
    ignore_unused_parameters: bool = False,
    **static_kwargs,
) -> Callable[..., Tensor]:
    raise NotImplementedError

    # merge_dictionaries_fn = functools.partial(
    #     _merge_dictionaries,
    #     ignore_unused_parameters=ignore_unused_parameters,
    # )
    #
    # def mapped_fn(
    #         positions: Tensor,
    #         bonds: Optional[Tensor] = None,
    #         kinds: Optional[Tensor] = None,
    #         **kwargs,
    # ) -> Tensor:
    #     accumulator = torch.tensor(
    #         0.0,
    #         device=positions.device,
    #         dtype=positions.dtype,
    #     )
    #
    #     distance_fn = functools.partial(displacement_fn, **kwargs)
    #
    #     distance_fn = torch.func.vmap(distance_fn, 0, 0)
    #
    #     if bonds is not None:
    #         parameters = merge_dictionaries_fn(static_kwargs, kwargs)
    #
    #         for name, parameter in parameters.items():
    #             if kinds is not None:
    #                 parameters[name] = _to_bond_kind_parameters(
    #                     parameter,
    #                     kinds,
    #                 )
    #
    #         interactions = distance_fn(
    #             positions[bonds[:, 0]],
    #             positions[bonds[:, 1]],
    #         )
    #
    #         interactions = _safe_sum(fn(interactions, **parameters))
    #
    #         accumulator = accumulator + interactions
    #
    #     if static_bonds is not None:
    #         parameters = merge_dictionaries_fn(static_kwargs, kwargs)
    #
    #         for name, parameter in parameters.items():
    #             if static_kinds is not None:
    #                 parameters[name] = _to_bond_kind_parameters(
    #                     parameter,
    #                     static_kinds,
    #                 )
    #
    #         interactions = distance_fn(
    #             positions[static_bonds[:, 0]],
    #             positions[static_bonds[:, 1]],
    #         )
    #
    #         interactions = _safe_sum(fn(interactions, **parameters))
    #
    #         accumulator = accumulator + interactions
    #
    #     return accumulator
    #
    # return mapped_fn


def _dihedral_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
):
    raise NotImplementedError


def _long_range_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
):
    raise NotImplementedError


def _angle_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
):
    raise NotImplementedError


def _mesh_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
):
    raise NotImplementedError


def _kwargs_to_neighbor_list_parameters(
    format: _NeighborListFormat,
    indices: Tensor,
    kinds: Tensor,
    kwargs: Dict[str, Tensor],
    combinators: Dict[str, Callable],
) -> Dict[str, Tensor]:
    r"""Converts keyword arguments to neighbor list parameters.

    Parameters
    ----------
    format : _NeighborListFormat
        The format of the neighbor list.

    indices : Tensor
        A tensor containing the indices of the neighbor candidates for each particle.
        The shape of the tensor is expected to be (n, m) where `n` is the number
        of particles and `m` is the number of neighbor candidates per particle.

    kinds : Tensor
        A tensor containing the kinds of particles. If None, the parameters will be
        treated as single-dimensional tensors.

    kwargs : dict
        A dictionary of keyword arguments where the keys are parameter names and the
        values are tensors representing the parameters.

    combinators : dict
        A dictionary of combinator functions where the keys are parameter names and the
        values are functions that combine two tensors.

    Returns
    -------
    output : dict
        A dictionary of neighbor list parameters where the keys are parameter names and
        the values are tensors representing the parameters.
    """
    parameters = {}

    for name, parameter in kwargs.items():
        if kinds is None or (isinstance(parameter, Tensor) and parameter.dim() == 1):
            combinator = combinators.get(name, lambda x, y: 0.5 * (x + y))

            parameters[name] = _to_neighbor_list_matrix_parameters(
                format,
                indices,
                parameter,
                combinator,
            )
        else:
            if name in combinators:
                raise ValueError

            parameters[name] = _to_neighbor_list_kind_parameters(
                format,
                indices,
                kinds,
                parameter,
            )

    return parameters


def _kwargs_to_pair_parameters(
    kwargs: Dict[str, Union[_ParameterTree, Tensor, float, PyTree]],
    combinators: Dict[str, Callable],
    kinds: Tensor | None = None,
) -> Dict[str, Tensor]:
    r"""Converts keyword arguments to pair interaction parameters.

    Parameters
    ----------
    kwargs : dict
        A dictionary of keyword arguments where the keys are parameter names and the
        values are instances of `_ParameterTree`, `Tensor`, `float`, or `PyTree`.

    combinators : dict
        A dictionary of combinator functions where the keys are parameter names and the
        values are functions that combine two tensors.

    kinds : Tensor, optional
        A tensor containing the kinds of particles. If None, the parameters will be
        treated as single-dimensional tensors.

    Returns
    -------
    output : dict
        A dictionary of pair interaction parameters where the keys are parameter names
        and the values are tensors representing the parameters.
    """
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


def _merge_dictionaries(
    this: Dict,
    that: Dict,
    ignore_unused_parameters: bool = False,
):
    r"""Merges two dictionaries, optionally ignoring unused parameters.

    Parameters
    ----------
    this : dict
        The first dictionary to be merged.

    that : dict
        The second dictionary to be merged.

    ignore_unused_parameters : bool, optional
        If True, only keys present in the first dictionary will be updated with values
        from the second dictionary. If False, all keys from both dictionaries will be
        included in the merged dictionary.

    Returns
    -------
    output : dict
        A dictionary containing the merged key-value pairs from both input dictionaries.
    """
    if not ignore_unused_parameters:
        return {**this, **that}

    merged_dictionaries = dict(this)

    for this_key in merged_dictionaries.keys():
        that_value = that.get(this_key)

        if that_value is not None:
            merged_dictionaries[this_key] = that_value

    return merged_dictionaries


def _neighbor_list_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    kinds: Tensor | None = None,
    dimension: Optional[Tuple[int, ...]] = None,
    ignore_unused_parameters: bool = False,
    **kwargs,
) -> Callable[..., Tensor]:
    """Creates a function to compute interactions based on a neighbor list.

    Parameters
    ----------
    fn : Callable[..., Tensor]
        The function to compute the interaction given the distances and other parameters.

    displacement_fn : Callable[[Tensor, Tensor], Tensor]
        A function to compute the displacement between two sets of positions.

    kinds : Tensor, optional
        A tensor containing the kinds of particles. If None, the parameters will be
        treated as single-dimensional tensors.

    dimension : tuple of int, optional
        The dimensions over which to sum the interactions. If None, the interactions
        will be summed over all dimensions.

    ignore_unused_parameters : bool, optional
        If True, only keys present in the first dictionary will be updated with values
        from the second dictionary. If False, all keys from both dictionaries will be
        included in the merged dictionary.

    **kwargs : dict
        Additional keyword arguments to be passed to the interaction function.

    Returns
    -------
    mapped_fn : Callable[..., Tensor]
        A function that computes the interactions based on the neighbor list.
    """
    parameters, combinators = {}, {}

    for name, parameter in kwargs.items():
        if isinstance(parameter, Callable):
            combinators[name] = parameter
        elif isinstance(parameter, tuple) and isinstance(parameter[0], Callable):
            assert len(parameter) == 2

            combinators[name], parameters[name] = parameter[0], parameter[1]
        else:
            parameters[name] = parameter

    merge_dictionaries = functools.partial(
        _merge_dictionaries,
        ignore_unused_parameters=ignore_unused_parameters,
    )

    def mapped_fn(
        position: Tensor,
        neighbor_list: _NeighborList,
        **dynamic_kwargs,
    ) -> Tensor:
        r"""Computes interactions based on a neighbor list and dynamic keyword arguments.

        Parameters
        ----------
        position : Tensor
            A tensor representing the positions of the particles in the system.
            The shape of the tensor is expected to be (n, d) where `n` is the number
            of particles and `d` is the dimensionality of the system.

        neighbor_list : _NeighborList
            An object containing the neighbor list, including indices, item size,
            maximum size, and other relevant information.

        **dynamic_kwargs : dict
            Additional dynamic keyword arguments to be passed to the distance and interaction functions.

        Returns
        -------
        output : Tensor
            A tensor representing the computed interactions based on the neighbor list.
        """
        distance_fn = functools.partial(displacement_fn, **dynamic_kwargs)

        _kinds = dynamic_kwargs.get("kinds", kinds)

        normalization = 2.0

        if is_neighbor_list_sparse(neighbor_list.format):
            distances = _map_bond(distance_fn)(
                safe_index(position, neighbor_list.indexes[0]),
                safe_index(position, neighbor_list.indexes[1]),
            )

            mask = torch.less(neighbor_list.indexes[0], position.shape[0])

            if neighbor_list.format is _NeighborListFormat.ORDERED_SPARSE:
                normalization = 1.0
        else:
            d = _map_neighbor(distance_fn)
            r_neigh = safe_index(position, neighbor_list.indexes)
            distances = d(position, r_neigh)

            mask = torch.less(neighbor_list.indexes, position.shape[0])

        merged_kwargs = merge_dictionaries(parameters, dynamic_kwargs)
        merged_kwargs = _kwargs_to_neighbor_list_parameters(
            neighbor_list.format,
            neighbor_list.indexes,
            _kinds,
            merged_kwargs,
            combinators,
        )

        out = fn(distances, **merged_kwargs)

        if out.ndim > mask.ndim:
            mask = torch.reshape(
                mask,
                [*mask.shape, *([1] * (out.ndim - mask.ndim))],
            )

        out = torch.where(mask, out, torch.tensor(0.0))

        if dimension is None:
            return torch.divide(_safe_sum(out), normalization)

        if 0 in dimension and 1 not in dimension:
            raise ValueError

        if not is_neighbor_list_sparse(neighbor_list.format):
            return torch.divide(_safe_sum(out, dimension=dimension), normalization)

        if 0 in dimension:
            return _safe_sum(out, dimension=tuple(a - 1 for a in dimension if a > 1))

        if neighbor_list.format is _NeighborListFormat.ORDERED_SPARSE:
            raise ValueError

        out = _safe_sum(out, dimension=tuple(a - 1 for a in dimension if a > 1))
        return torch.divide(
            _segment_sum(
                out,
                neighbor_list.indexes[0],
                position.shape[0],
            ),
            normalization,
        ).to(dtype=position.dtype)

    return mapped_fn


def _pair_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    kinds: Optional[Union[int, Tensor]] = None,
    dimension: Optional[Tuple[int, ...]] = None,
    keepdim: bool = False,
    ignore_unused_parameters: bool = False,
    **kwargs,
) -> Callable[..., Tensor]:
    r"""Creates a function to compute pairwise interactions between particles.

    Parameters
    ----------
    fn : Callable[..., Tensor]
        The function to compute the interaction given the distances and other parameters.

    displacement_fn : Callable[[Tensor, Tensor], Tensor]
        A function to compute the displacement between two sets of positions.

    kinds : int or Tensor, optional
        An integer or tensor representing the kinds of particles. If None, the parameters
        will be treated as single-dimensional tensors.

    dimension : tuple of int, optional
        The dimensions over which to sum the interactions. If None, the interactions
        will be summed over all dimensions.

    keepdim : bool, optional
        If True, retains reduced dimensions with length 1.

    ignore_unused_parameters : bool, optional
        If True, only keys present in the first dictionary will be updated with values
        from the second dictionary. If False, all keys from both dictionaries will be
        included in the merged dictionary.

    **kwargs : dict
        Additional keyword arguments to be passed to the interaction function.

    Returns
    -------
    mapped_fn : Callable[..., Tensor]
        A function that computes the pairwise interactions based on the provided parameters.
    """
    parameters, combinators = {}, {}

    for name, parameter in list(kwargs.items()):
        if isinstance(parameter, Callable):
            combinators[name] = parameter
            del kwargs[name]

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
            r"""Computes pairwise interactions for particles without kinds.

            Parameters
            ----------
            _position : Tensor
                A tensor representing the positions of the particles in the system.
                The shape of the tensor is expected to be (n, d) where `n` is the number
                of particles and `d` is the dimensionality of the system.

            **_dynamic_kwargs : dict
                Additional dynamic keyword arguments to be passed to the distance and interaction functions.

            Returns
            -------
            output : Tensor
                A tensor representing the computed pairwise interactions.
            """
            distance_fn = functools.partial(displacement_fn, **_dynamic_kwargs)

            distances = map_product(distance_fn)(_position, _position)

            dictionaries = merge_dicts(parameters, _dynamic_kwargs)

            to_parameters = _kwargs_to_pair_parameters(
                dictionaries,
                combinators,
            )

            interaction = fn(distances, **to_parameters)

            interaction = _zero_diagonal_mask(interaction)

            interaction = _safe_sum(
                interaction, dimension=dimension, keep_dimension=keepdim
            )

            return interaction * 0.5

        return mapped_fn

    if isinstance(kinds, Tensor):
        if not isinstance(kinds, Tensor) or kinds.is_floating_point():
            raise ValueError

        kinds_count = int(torch.max(kinds))

        if dimension is not None or keepdim:
            raise ValueError

        def mapped_fn(_position: Tensor, **_dynamic_kwargs):
            r"""Computes pairwise interactions for particles with kinds.

            Parameters
            ----------
            _position : Tensor
                A tensor representing the positions of the particles in the system.
                The shape of the tensor is expected to be (n, d) where `n` is the number
                of particles and `d` is the dimensionality of the system.

            **_dynamic_kwargs : dict
                Additional dynamic keyword arguments to be passed to the distance and interaction functions.

            Returns
            -------
            output : Tensor
                A tensor representing the computed pairwise interactions.
            """
            interaction = torch.tensor(0.0, dtype=torch.float32)

            distance_fn = functools.partial(displacement_fn, **_dynamic_kwargs)

            distance_fn = map_product(distance_fn)

            for m in range(kinds_count + 1):
                for n in range(m, kinds_count + 1):
                    distance = distance_fn(
                        _position[kinds == m],
                        _position[kinds == n],
                    )

                    _kwargs = merge_dicts(parameters, _dynamic_kwargs)

                    s_kwargs = _kwargs_to_pair_parameters(_kwargs, combinators, (m, n))

                    y = fn(distance, **s_kwargs)

                    if m == n:
                        y = _zero_diagonal_mask(y)

                        y = _safe_sum(y)

                        interaction = interaction + y * 0.5
                    else:
                        y = _safe_sum(y)

                        interaction = interaction + y

            return interaction

        return mapped_fn

    if isinstance(kinds, int):
        kinds_count = kinds

        def mapped_fn(_position: Tensor, _kinds: Tensor, **_dynamic_kwargs):
            r"""Computes pairwise interactions for particles with a specified number of kinds.

            Parameters
            ----------
            _position : Tensor
                A tensor representing the positions of the particles in the system.
                The shape of the tensor is expected to be (n, d) where `n` is the number
                of particles and `d` is the dimensionality of the system.

            _kinds : Tensor
                A tensor representing the kinds of particles.

            **_dynamic_kwargs : dict
                Additional dynamic keyword arguments to be passed to the distance and interaction functions.

            Returns
            -------
            output : Tensor
                A tensor representing the computed pairwise interactions.
            """
            if not isinstance(_kinds, Tensor) or _kinds.is_floating_point():
                raise ValueError

            interaction = torch.tensor(0.0, dtype=torch.float32)

            num_particles = _position.shape[0]

            distance_fn = functools.partial(displacement_fn, **_dynamic_kwargs)

            distance_fn = map_product(distance_fn)

            _kwargs = merge_dicts(parameters, _dynamic_kwargs)

            distance = distance_fn(_position, _position)

            for m in range(kinds_count):
                for n in range(kinds_count):
                    a = torch.reshape(
                        _kinds == m,
                        [
                            num_particles,
                        ],
                    )
                    b = torch.reshape(
                        _kinds == n,
                        [
                            num_particles,
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

                    y = _safe_sum(y, dimension=dimension, keep_dimension=keepdim)

                    interaction = interaction + y

            return interaction / 2.0

        return mapped_fn

    raise ValueError


def _to_bond_kind_parameters(
    parameter: Tensor | _ParameterTree,
    kinds: Tensor,
) -> Tensor | _ParameterTree:
    r"""Converts parameters to bond kind parameters based on particle kinds.

    Parameters
    ----------
    parameter : Tensor or _ParameterTree
        The parameter to be converted. It can be a tensor or an instance of `_ParameterTree`.

    kinds : Tensor
        A tensor containing the kinds of particles. The shape of the tensor is expected
        to be (n,) where `n` is the number of particles.

    Returns
    -------
    output : Tensor or _ParameterTree
        The converted bond kind parameters. The output type matches the input type.
    """
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


def _to_neighbor_list_kind_parameters(
    format: _NeighborListFormat,
    indices: Tensor,
    kinds: Tensor,
    parameters: _ParameterTree | Tensor | float,
) -> PyTree | _ParameterTree | Tensor | float:
    r"""Safely computes the sum of elements in a tensor along a specified
    dimension, promoting the data type to avoid precision loss.

    Parameters
    ----------
    format : _NeighborListFormat
        An enumeration representing the format of a neighbor list. Could be
        0 (Dense), 1 (Ordered sparse), 2 (Sparse).

    indices : Tensor
        A tensor containing the indexes of neighbors.

    kinds : Tensor
        Atomic labelling that contain atomic species, properties, or other
        metadata.

    parameters : _ParameterTree | Tensor | float
        parameters to be used for neighborlist interaction functions.

    Returns
    -------
    PyTree | _ParameterTree | Tensor | float
        The formatted parameters.
    """
    fn = functools.partial(
        lambda p, a, b: p[a, b],
        parameters,
    )

    match parameters:
        case parameters if isinstance(parameters, Tensor):
            match len(parameters.shape):
                case 0:
                    return parameters
                case 2:
                    if is_neighbor_list_sparse(format):
                        return manual_vmap(fn, (0, 0), 0)(
                            safe_index(kinds, indices[0]),
                            safe_index(kinds, indices[1]),
                        )

                    return manual_vmap(
                        manual_vmap(
                            fn,
                            in_dimension=(None, 0),
                        ),
                    )(kinds, safe_index(kinds, indices))
                case _:
                    raise ValueError
        case parameters if isinstance(parameters, _ParameterTree):
            match parameters.kind:
                case _ParameterTreeKind.KINDS:
                    # TODO (isaacsoh) remove lookup
                    def lookup(p, species_a, species_b):
                        return p[species_a, species_b]

                    if is_neighbor_list_sparse(format):
                        return optree.tree_map(
                            lambda parameter: manual_vmap(
                                lambda a, b: lookup(parameter, a, b), (0, 0), 0
                            )(
                                safe_index(kinds, indices[0]),
                                safe_index(kinds, indices[1]),
                            ),
                            parameters.tree,
                        )

                    return optree.tree_map(
                        lambda parameter: manual_vmap(
                            manual_vmap(
                                lambda a, b: lookup(parameter, a, b),
                                (None, 0),
                            )
                        )(
                            kinds,
                            safe_index(kinds, indices),
                        ),
                        parameters.tree,
                    )
                case _ParameterTreeKind.SPACE:
                    return parameters.tree
        case _:
            raise ValueError

    return parameters


def manual_vmap(
    func: Callable,
    in_dimension: Union[int, Tuple[Union[int, None], ...]] = 0,
    out_dimension: Union[int, Tuple[int, ...]] = 0,
    randomness: str = "error",
    *,
    chunk_size: Union[None, int] = None,
) -> Callable:
    r"""Manually vectorizes a function over a specified batch dimension.

    This function replaces `torch.vmap` and allows for manual batching of inputs
    to a function. It iterates over the batch dimension, applies the function to
    each slice, and stacks the results along the specified output dimension.

    Parameters
    ----------
    func : Callable
        The function to be vectorized.

    in_dimension : int or tuple of int or None, optional
        The dimensions of the inputs to be batched. If an integer, it specifies
        the batch dimension for all inputs. If a tuple, it specifies the batch
        dimension for each input individually. If None, the input is not batched.

    out_dimension : int or tuple of int, optional
        The dimensions along which to stack the outputs. If an integer, it specifies
        the output dimension for all outputs. If a tuple, it specifies the output
        dimension for each output individually.

    randomness : str, optional
        Specifies how to handle randomness. Currently, only "error" is supported,
        which raises an error if randomness is encountered.

    chunk_size : int or None, optional
        The size of chunks to process at a time. If None, the entire batch is processed
        at once.

    Returns
    -------
    batched_func : Callable
        A function that applies `func` to batched inputs and stacks the results.

    Notes
    -----
    This function is a manual replacement for `torch.vmap` and is useful when
    `torch.vmap` is not available or when custom batching behavior is required.
    """

    def batched_func(*args, **kwargs):
        if isinstance(in_dimension, int):
            batch_size = args[0].shape[in_dimension]
        else:
            batch_size = next(
                arg.shape[dim]
                for arg, dim in zip(args, in_dimension)
                if dim is not None
            )

        results = []

        for i in range(batch_size):
            sliced_args = []
            for arg, dim in zip(
                args,
                in_dimension
                if isinstance(in_dimension, tuple)
                else [in_dimension] * len(args),
            ):
                if dim is None:
                    sliced_args.append(arg)
                else:
                    sliced_args.append(arg.select(dim, i))

            result = func(*sliced_args, **kwargs)

            results.append(result)

        if isinstance(out_dimension, int):
            return torch.stack(results, dim=out_dimension)
        else:
            return tuple(
                torch.stack([res[i] for res in results], dim=out_dimension[i])
                for i in range(len(results[0]))
            )

    return batched_func


def _to_neighbor_list_matrix_parameters(
    format: _NeighborListFormat,
    indices: Tensor,
    parameters: _ParameterTree | Tensor | float,
    combinator: Callable[[Tensor, Tensor], Tensor],
) -> PyTree | _ParameterTree | Tensor | float:
    r"""Converts parameters to neighbor list matrix parameters based on the format and indices.

    Parameters
    ----------
    format : _NeighborListFormat
        The format of the neighbor list.

    indices : Tensor
        A tensor containing the indices of the neighbor candidates for each particle.
        The shape of the tensor is expected to be (n, m) where `n` is the number
        of particles and `m` is the number of neighbor candidates per particle.

    parameters : _ParameterTree, Tensor, or float
        The parameters to be converted. It can be an instance of `_ParameterTree`, a tensor,
        or a float.

    combinator : Callable[[Tensor, Tensor], Tensor]
        A function that combines two tensors.

    Returns
    -------
    output : PyTree, _ParameterTree, Tensor, or float
        The converted neighbor list matrix parameters. The output type matches the input type.
    """
    match parameters:
        case parameters if isinstance(parameters, Tensor):
            match parameters.ndim:
                case 0:
                    return parameters
                case 1:
                    if is_neighbor_list_sparse(format):
                        return manual_vmap(combinator, (0, 0), 0)(
                            safe_index(parameters, indices[0]),
                            safe_index(parameters, indices[1]),
                        )

                    return combinator(
                        parameters[:, None],
                        safe_index(parameters, indices),
                    )
                case 2:
                    if is_neighbor_list_sparse(format):
                        displacement = lambda a, b: safe_index(parameters, a, b)
                        return manual_vmap(displacement, (0, 0), 0)(
                            indices[0],
                            indices[1],
                        )

                    return manual_vmap(
                        manual_vmap(
                            lambda a, b: safe_index(parameters, a, b),
                            (None, 0),
                        ),
                    )(
                        torch.arange(indices.shape[0], dtype=torch.int32),
                        indices,
                    )
                case _:
                    raise ValueError
        case parameters if isinstance(parameters, _ParameterTree):
            match parameters.kind:
                case _ParameterTreeKind.BOND:
                    if is_neighbor_list_sparse(format):
                        return optree.tree_map(
                            lambda parameter: manual_vmap(
                                functools.partial(
                                    lambda p, a, b: safe_index(p, a, b),
                                    parameter,
                                ),
                                (0, 0),
                                0,
                            )(
                                indices[0],
                                indices[1],
                            ),
                            parameters.tree,
                        )

                    return optree.tree_map(
                        lambda parameter: manual_vmap(
                            manual_vmap(
                                functools.partial(safe_index, parameter),
                                (None, 0),
                            ),
                        )(
                            torch.arange(indices.shape[0], dtype=torch.int32),
                            indices,
                        ),
                        parameters.tree,
                    )
                case _ParameterTreeKind.PARTICLE:
                    if is_neighbor_list_sparse(format):
                        return optree.tree_map(
                            lambda parameter: manual_vmap(combinator, (0, 0), 0)(
                                safe_index(parameter, indices[0]),
                                safe_index(parameter, indices[1]),
                            ),
                            parameters.tree,
                        )

                    return optree.tree_map(
                        lambda parameter: _map_neighbor(
                            combinator,
                        )(
                            parameter,
                            safe_index(parameter, indices),
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


def interact(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    interaction: Literal[
        "angle",
        "bond",
        "dihedral",
        "long-range",
        "mesh",
        "neighbor_list",
        "pair",
    ],
    *,
    bonds: Optional[Tensor] = None,
    kinds: Optional[Union[int, Tensor]] = None,
    dimension: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    ignore_unused_parameters: bool = False,
    **kwargs,
) -> Callable[..., Tensor]:
    r"""
    Define interactions between elements of a system.

    For a collection of $N$ elements, $\vec{r}_i \in \mathbb{R}^{D}$, where $1 \leq i \leq N$, energy is the function $U : \mathbb{R}^{N \times D} \rightarrow \mathbb{R}$. Energy is used by a simulation by applying Newton's laws: $m \vec{\ddot{r}}_{i} = - \nabla_{\vec{r}_{i}} U$ where $m$ is mass. Rather than defining an energy as an interaction between all the elements in the simulation space simultaneously, it's preferable to use a pairwise energy function based on the displacement between a pair of elements, $u(\vec{r}_{i} - \vec{r}_{j})$. Total energy is defined by the sum over pairwise interactions:

    $$U = \frac{1}{2} \sum_{i \neq j} u(\vec{r}_{i} - \vec{r}_{j}).$$

    To facilitate the construction of functions from interactions, `interact` returns a function to map bonds, neighbors, pairs, or triplets interactions and transforms them to operate on an entire simulation.

    Parameters
    ----------
    fn : Callable[..., Tensor]
        Function that takes distances or displacements of shape `(n, m)` or `(n, m, spatial_dimension)` and `kwargs` and returns values of shape `(n, m, spatial_dimension)`. The function must be a differentiable function as the force is computed using automatic differentiation (see `prescient.func.force`).

    displacement_fn : Callable[[Tensor, Tensor], Tensor]
        Displacement function that takes positions of shape `(spatial_dimension)` and `(spatial_dimension)` and returns distances or displacements of shape `()` or `(spatial_dimension)`.

    interaction : Literal["bond", "neighbor_list", "pair", "triplet"]
        One of the following types of interactions:

        -   `"angle"`,

        -   `"bond"`, transforms a function that acts on a single pair of elements to a function that acts on a set of bonds.

        -  `"dihedral"`,

        -   `"long-range"`,

        -   `"neighbor_list"`, transforms a function that acts on pairs of elements to a function that acts on neighbor lists.

        -   `"pair"`, transforms a function that acts on a pair of elements to a function that acts on a system of interacting elements.

        -   `"triplet"`, transforms a function that acts on triplets of elements to a function that acts on a system of interacting elements. Many common empirical potentials include three-body terms, this type of pairwise interaction simplifies the loss computation by transforming a loss function that acts on two pairwise displacements or distances to a loss function that acts on a system of interacting elements.

    bonds : Optional[Tensor], default=None

    kinds : Optional[Tensor], default=None
        Kinds for the different elements. Should either be `None` (in which case it is assumed that all the elements have the same kind) or labels of shape `(n)`. If `intraction` is `"pair"` or `"triplet"`, kinds can be dynamically specified by passing the `kinds` keyword argument to the mapped function.

    dimension : Optional[Union[int, Tuple[int, ...]]], default=None
        Dimension or dimensions to reduce. If `None`, all dimensions are reduced.

    keepdim : bool, default=False
        Whether the output has `dim` retained or not.

    ignore_unused_parameters : bool, default=True

    kwargs :
        `kwargs` passed to the function. Depends on the `interaction` type:

        *   If `interaction` is `"bond"` and `kinds` is `None`, must be a scalar or a tensor of shape `(n)`. If `interaction` is `"bond"` and `kinds` is not `None`, must be a scalar, a tensor of shape `(kinds)`, or a PyTree of parameters and corresponding mapping.

        *   If `interaction` is `"neighbor"` and `kinds` is `None`, must be a scalar, tensor of shape `(n)`, tensor of shape `(n, n)`, a PyTree of parameters and corresponding mapping, or a binary function that determines how per-element parameters are combined. If `kinds` is `None`, `kinds` is defined as the average of the two per-element parameters. If `interaction` is `"neighbor"` and `kinds` is not `None`, must be a scalar, a tensor of shape `(kinds, kinds)`, or a PyTree of parameters and corresponding mapping.

        *   If `interaction` is `"pair"` and `kinds` is `None`, must be a scalar, tensor of shape `(n)`, tensor of shape `(n, n)`, a PyTree of parameters and corresponding mapping, or a binary function that determines how per-element parameters are combined. If `kinds` is `None`, `kinds` is defined as the average of the two per-element parameters. If `interaction` is `"pair"` and `kinds` is not `None`, must be a scalar, a tensor of shape `(kinds, kinds)`, or a PyTree of parameters and corresponding mapping.

        *   If `interaction` is `"triplet"` and `kinds` is `None`, must be a scalar, tensor of shape `(n)` based on the central element, or a tensor of shape `(n, n, n)` defining triplet interactions. If `interaction` is `"triplet"` and `kinds` is not `None`, must be a scalar, a tensor of shape `(kinds)`, or a tensor of shape `(kinds, kinds, kinds)` defining triplet interactions.

    Returns
    -------
    : Callable[..., Tensor]
        Signature of the return function depends on `interaction`:

        *   `"bond"`:
                `(positions, bonds, kinds) -> Tensor`

            The return function can optionally take the keyword arguments `bonds` and `kinds` to dynamically allocate bonds.

        *   `"neighbor"`:
                `(positions, neighbors) -> Tensor`

            The return function takes positions of shape `(n, spatial_dimension)` and neighbor labels of shape `(n, neighbors)`.

        *   `"pair"`:
                `(positions, kinds, maximum_kind, **kwargs) -> Tensor`

            If `kinds` is `None` or static, the return function takes positions of shape `(n, spatial_dimension)`. If `kinds` is dynamic, the return function takes positions of shape `(n, spatial_dimension)`, integer labels of shape (n), and an integer specifying the maximum kind. The return function can optionally take keyword arguments to pass to the displacement function.

        *   `"triplet"`:
                `(positions, kinds, maximum_kind, **kwargs) -> Tensor`

            If `kinds` is `None` or static, the return function takes positions of shape `(n, spatial_dimension)`. If `kinds` is dynamic, the return function takes positions of shape `(n, spatial_dimension)`, integer labels of shape (n), and an integer specifying the maximum kind. The return function can optionally take keyword arguments to pass to the displacement function.

    Examples
    --------
    Create a pairwise interaction from a potential function:

        def fn(x: Tensor, a: float, e: float, s: float, **_) -> Tensor:
            return e / a * (1.0 - x / s) ** a

        displacement_fn, _ = prescient.func.space([10.0], parallelpiped=False)

        fn = prescient.func.interact(
            fn,
            displacement_fn,
            interaction="pair",
            a=2.0,
            e=1.0,
            s=1.0,
        )
    """
    match interaction:
        case "angle":
            return _angle_interaction(
                fn,
                displacement_fn,
            )
        case "bond":
            return _bond_interaction(
                fn,
                displacement_fn,
                static_bonds=bonds,
                static_kinds=kinds,
                ignore_unused_parameters=ignore_unused_parameters,
                **kwargs,
            )
        case "dihedral":
            return _dihedral_interaction(
                fn,
                displacement_fn,
            )
        case "long-range":
            return _long_range_interaction(
                fn,
                displacement_fn,
            )
        case "mesh":
            return _mesh_interaction(
                fn,
                displacement_fn,
            )
        case "neighbor_list":
            return _neighbor_list_interaction(
                fn,
                displacement_fn,
                kinds=kinds,
                dimension=dimension,
                ignore_unused_parameters=ignore_unused_parameters,
                **kwargs,
            )
        case "pair":
            return _pair_interaction(
                fn,
                displacement_fn,
                kinds=kinds,
                dimension=dimension,
                keepdim=keepdim,
                ignore_unused_parameters=ignore_unused_parameters,
                **kwargs,
            )
        case _:
            raise ValueError
