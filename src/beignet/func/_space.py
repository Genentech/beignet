from typing import Callable, TypeVar

import torch
from torch import Tensor

import beignet

T = TypeVar("T")


def space(
    box: Tensor | None = None,
    *,
    normalized: bool = True,
    parallelepiped: bool = True,
    remapped: bool = True,
) -> (Callable, Callable):
    r"""Define a simulation space.

    This function is fundamental in constructing simulation spaces derived from
    subsets of $\mathbb{R}^{D}$ (where $D = 1$, $2$, or $3$) and is
    instrumental in setting up simulation environments with specific
    characteristics (e.g., periodic boundary conditions). The function returns
    a displacement function and a shift function to compute particle
    interactions and movements in space.

    This function supports deformation of the simulation cell, crucial for
    certain types of simulations, such as those involving finite deformations
    or the computation of elastic constants.

    Parameters
    ----------
    box : Tensor | None, default=None
        Interpretation varies based on the value of `parallelepiped`. If
        `parallelepiped` is `True`, must be an affine transformation, $T$,
        specified in one of three ways:

        1. a cube, $L$;
        2. an orthorhombic unit cell, $[L_{x}, L_{y}, L_{z}]$; or
        3. a triclinic cell, upper triangular matrix.

        If `parallelepiped` is `False`, must be the edge lengths.

        If `box` is `None`, the simulation space has free boundary conditions.

    normalized : bool, default=True
        If `normalized` is `True`, positions are stored in the unit cube.
        Displacements and shifts are computed in a normalized simulation space
        and can be transformed back to real simulation space using the affine
        transformation. If `normalized` is `False`, positions are expressed and
        computations performed directly in the real simulation space.

    parallelepiped : bool, default=True
        If `True`, the simulation space is defined as a ${1, 2, 3}$-dimensional
        parallelepiped with periodic boundary conditions. If `False`, the space
        is defined on a ${1, 2, 3}$-dimensional hypercube.

    remapped : bool, default=True
        If `True`, positions and displacements are remapped to stay in the
        bounds of the defined simulation space. A rempapped simulation space is
        topologically equivalent to a torus, ensuring that particles exiting
        one boundary re-enter from the opposite side.

    Returns
    -------
    (Callable[[Tensor, Tensor], Tensor], Callable[[Tensor, Tensor], Tensor])
        A pair of functions:

        1.  The displacement function, $\vec{d}$, measures the
            difference between two points in the simulation space, factoring in
            the geometry and boundary conditions. This function is used to
            calculate particle interactions and dynamics.

        2.  The shift function, $u$, applies a displacement vector to a point
            in the space, effectively moving it. This function is used to
            update simulated particle positions.

    Examples
    --------
        transformation = torch.tensor([10.0])

        displacement_fn, shift_fn = space(
            transformation,
            normalized=False,
        )

        normalized_displacement_fn, normalized_shift_fn = space(
            transformation,
            normalized=True,
        )

        normalized_position = torch.rand([4, 3])

        position = transformation * normalized_position

        displacement = torch.randn([4, 3])

        torch.testing.assert_close(
            displacement_fn(position[0], position[1]),
            normalized_displacement_fn(
                normalized_position[0],
                normalized_position[1],
            ),
        )
    """
    if isinstance(box, (int, float)):
        box = torch.tensor([box])

    if box is None:

        def displacement_fn(
            input: Tensor,
            other: Tensor,
            *,
            perturbation: Tensor | None = None,
            **_,
        ) -> Tensor:
            if len(input.shape) != 1:
                raise ValueError

            if input.shape != other.shape:
                raise ValueError

            if perturbation is not None:
                transform = input - other

                match transform.ndim:
                    case 0:
                        return perturbation * transform
                    case 1:
                        return torch.einsum(
                            "i,...i->...i",
                            transform,
                            perturbation,
                        )
                    case 2:
                        return torch.einsum(
                            "ij,...j->...i",
                            transform,
                            perturbation,
                        )
                    case _:
                        raise ValueError

            return input - other

        def shift_fn(input: Tensor, other: Tensor, **_) -> Tensor:
            return input + other

        return displacement_fn, shift_fn

    if parallelepiped:
        inverted_transform = beignet.invert_transform(box)

        if normalized:

            def displacement_fn(
                input: Tensor,
                other: Tensor,
                *,
                perturbation: Tensor | None = None,
                **kwargs,
            ) -> Tensor:
                _transform = box

                _inverted_transform = inverted_transform

                if "transform" in kwargs:
                    _transform = kwargs["transform"]

                if "updated_transform" in kwargs:
                    _transform = kwargs["updated_transform"]

                if len(input.shape) != 1:
                    raise ValueError

                if input.shape != other.shape:
                    raise ValueError

                displacement = beignet.apply_transform(
                    torch.remainder(input - other + 1.0 * 0.5, 1.0) - 1.0 * 0.5,
                    _transform,
                )

                if perturbation is not None:
                    match displacement.ndim:
                        case 0:
                            return perturbation * displacement
                        case 1:
                            return torch.einsum(
                                "i,...i->...i",
                                displacement,
                                perturbation,
                            )
                        case 2:
                            return torch.einsum(
                                "ij,...j->...i",
                                displacement,
                                perturbation,
                            )
                        case _:
                            raise ValueError

                return displacement

            if remapped:

                def u(input: Tensor, other: Tensor) -> Tensor:
                    return torch.remainder(input + other, 1.0)

                def shift_fn(input: Tensor, other: Tensor, **kwargs) -> Tensor:
                    _transform = box

                    _inverted_transform = inverted_transform

                    if "transform" in kwargs:
                        _transform = kwargs["transform"]

                        _inverted_transform = beignet.invert_transform(_transform)

                    if "updated_transform" in kwargs:
                        _transform = kwargs["updated_transform"]

                    return u(input, beignet.apply_transform(other, _inverted_transform))

                return displacement_fn, shift_fn

            def shift_fn(input: Tensor, other: Tensor, **kwargs) -> Tensor:
                _transform = box

                _inverted_transform = inverted_transform

                if "transform" in kwargs:
                    _transform = kwargs["transform"]

                    _inverted_transform = beignet.invert_transform(_transform)

                if "updated_transform" in kwargs:
                    _transform = kwargs["updated_transform"]

                return input + beignet.apply_transform(other, _inverted_transform)

            return displacement_fn, shift_fn

        def displacement_fn(
            input: Tensor,
            other: Tensor,
            *,
            perturbation: Tensor | None = None,
            **kwargs,
        ) -> Tensor:
            _transform = box

            _inverted_transform = inverted_transform

            if "transform" in kwargs:
                _transform = kwargs["transform"]

                _inverted_transform = beignet.invert_transform(_transform)

            if "updated_transform" in kwargs:
                _transform = kwargs["updated_transform"]

            input = beignet.apply_transform(input, _inverted_transform)
            other = beignet.apply_transform(other, _inverted_transform)

            if len(input.shape) != 1:
                raise ValueError

            if input.shape != other.shape:
                raise ValueError

            displacement = beignet.apply_transform(
                torch.remainder(input - other + 1.0 * 0.5, 1.0) - 1.0 * 0.5,
                _transform,
            )

            if perturbation is not None:
                match displacement.ndim:
                    case 0:
                        return perturbation * displacement
                    case 1:
                        return torch.einsum(
                            "i,...i->...i",
                            displacement,
                            perturbation,
                        )
                    case 2:
                        return torch.einsum(
                            "ij,...j->...i",
                            displacement,
                            perturbation,
                        )
                    case _:
                        raise ValueError

            return displacement

        if remapped:

            def u(input: Tensor, other: Tensor) -> Tensor:
                return torch.remainder(input + other, 1.0)

            def shift_fn(input: Tensor, other: Tensor, **kwargs) -> Tensor:
                _transform = box

                _inverted_transform = inverted_transform

                if "transform" in kwargs:
                    _transform = kwargs["transform"]

                    _inverted_transform = beignet.invert_transform(
                        _transform,
                    )

                if "updated_transform" in kwargs:
                    _transform = kwargs["updated_transform"]

                return beignet.apply_transform(
                    u(
                        beignet.apply_transform(_inverted_transform, input),
                        beignet.apply_transform(_inverted_transform, other),
                    ),
                    _transform,
                )

            return displacement_fn, shift_fn

        def shift_fn(input: Tensor, other: Tensor, **_) -> Tensor:
            return input + other

        return displacement_fn, shift_fn

    def displacement_fn(
        input: Tensor,
        other: Tensor,
        *,
        perturbation: Tensor | None = None,
        **_,
    ) -> Tensor:
        if len(input.shape) != 1:
            raise ValueError

        if input.shape != other.shape:
            raise ValueError

        displacement = torch.remainder(
            input - other + box * 0.5,
            box,
        )

        if perturbation is not None:
            transform = displacement - box * 0.5

            match transform.ndim:
                case 0:
                    return perturbation * transform
                case 1:
                    return torch.einsum(
                        "i,...i->...i",
                        transform,
                        perturbation,
                    )
                case 2:
                    return torch.einsum(
                        "ij,...j->...i",
                        transform,
                        perturbation,
                    )
                case _:
                    raise ValueError

        return displacement - box * 0.5

    if remapped:

        def shift_fn(input: Tensor, other: Tensor, **_) -> Tensor:
            return torch.remainder(input + other, box)
    else:

        def shift_fn(input: Tensor, other: Tensor, **_) -> Tensor:
            return input + other

    return displacement_fn, shift_fn
