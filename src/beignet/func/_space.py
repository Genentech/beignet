from typing import Callable, TypeVar

import torch
from torch import Tensor
from torch.autograd import Function

T = TypeVar("T")


def _inverse_transform(transformation: Tensor) -> Tensor:
    """
    Calculates the inverse of an affine transformation matrix.

    Parameters
    ----------
    transformation : Tensor
        The affine transformation matrix to be inverted.

    Returns
    -------
    Tensor
        The inverse of the given affine transformation matrix.
    """
    if transformation.ndim in {0, 1}:
        return 1.0 / transformation

    if transformation.ndim == 2:
        return torch.linalg.inv(transformation)

    raise ValueError("Unsupported transformation dimensions.")


def _apply_transform(transformation: Tensor, position: Tensor) -> Tensor:
    """
    Applies an affine transformation to the position vector.

    Parameters
    ----------
    position : Tensor
        Position, must have the shape `(..., dimension)`.

    transformation : Tensor
        The affine transformation matrix, must be a scalar, a vector, or a
        matrix with the shape `(dimension, dimension)`.

    Returns
    -------
    Tensor
        Affine transformed position vector, has the same shape as the
        position vector.
    """
    if transformation.ndim == 0:
        return position * transformation

    indices = [chr(ord("a") + index) for index in range(position.ndim - 1)]

    indices = "".join(indices)

    if transformation.ndim == 1:
        return torch.einsum(
            f"i,{indices}i->{indices}i",
            transformation,
            position,
        )

    if transformation.ndim == 2:
        return torch.einsum(
            f"ij,{indices}j->{indices}i",
            transformation,
            position,
        )

    raise ValueError("Unsupported transformation dimensions.")


def apply_transform(transformation: Tensor, position: Tensor) -> Tensor:
    """
    Return affine transformed position.

    Parameters
    ----------
    transformation : Tensor
        Affine transformation matrix, must have shape
        `(dimension, dimension)`.

    position : Tensor
        Position, must have shape `(..., dimension)`.

    Returns
    -------
    Tensor
        Affine transformed position of shape `(..., dimension)`.
    """

    class _Transform(Function):
        generate_vmap_rule = True

        @staticmethod
        def forward(transformation: Tensor, position: Tensor) -> Tensor:
            """
            Return affine transformed position.

            Parameters
            ----------
            transformation : Tensor
                Affine transformation matrix, must have shape
                `(dimension, dimension)`.

            position : Tensor
                Position, must have shape `(..., dimension)`.

            Returns
            -------
            Tensor
                Affine transformed position of shape `(..., dimension)`.
            """
            return _apply_transform(transformation, position)

        @staticmethod
        def setup_context(ctx, inputs, output):
            transformation, position = inputs

            ctx.save_for_backward(transformation, position, output)

        @staticmethod
        def jvp(
            ctx,
            grad_transformation: Tensor,
            grad_position: Tensor,
        ) -> (Tensor, Tensor):
            transformation, position, _ = ctx.saved_tensors

            output = _apply_transform(transformation, position)

            grad_output = grad_position + _apply_transform(
                grad_transformation,
                position,
            )

            return output, grad_output

        @staticmethod
        def backward(ctx, grad_output: Tensor) -> (Tensor, Tensor):
            _, _, output = ctx.saved_tensors

            return output, grad_output

    return _Transform.apply(transformation, position)


def space(
    dimensions: Tensor | None = None,
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
    a a displacement function and a shift function to compute particle
    interactions and movements in space.

    This function supports deformation of the simulation cell, crucial for
    certain types of simulations, such as those involving finite deformations
    or the computation of elastic constants.

    Parameters
    ----------
    dimensions : Tensor | None, default=None
        Dimensions of the simulation space.

        Interpretation varies based on the value of `parallelepiped`. If
        `parallelepiped` is `True`, must be an affine transformation, $T$,
        specified in one of three ways:

        1. a cube, $L$;
        2. an orthorhombic unit cell, $[L_{x}, L_{y}, L_{z}]$; or
        3. a triclinic cell, upper triangular matrix.

        If `parallelepiped` is `False`, must be the edge lengths.

        If `dimensions` is `None`, the simulation space has free boundary
        conditions.

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
    if isinstance(dimensions, (int, float)):
        dimensions = torch.tensor([dimensions])

    if dimensions is None:

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
                return _apply_transform(input - other, perturbation)

            return input - other

        def shift_fn(input: Tensor, other: Tensor, **_) -> Tensor:
            return input + other

        return displacement_fn, shift_fn

    if parallelepiped:
        inverse_transformation = _inverse_transform(dimensions)

        if normalized:

            def displacement_fn(
                input: Tensor,
                other: Tensor,
                *,
                perturbation: Tensor | None = None,
                **kwargs,
            ) -> Tensor:
                _transformation = dimensions

                _inverse_transformation = inverse_transformation

                if "transformation" in kwargs:
                    _transformation = kwargs["transformation"]

                if "updated_transformation" in kwargs:
                    _transformation = kwargs["updated_transformation"]

                if len(input.shape) != 1:
                    raise ValueError

                if input.shape != other.shape:
                    raise ValueError

                displacement = apply_transform(
                    _transformation,
                    torch.remainder(input - other + 1.0 * 0.5, 1.0) - 1.0 * 0.5,
                )

                if perturbation is not None:
                    return _apply_transform(displacement, perturbation)

                return displacement

            if remapped:

                def u(input: Tensor, other: Tensor) -> Tensor:
                    return torch.remainder(input + other, 1.0)

                def shift_fn(input: Tensor, other: Tensor, **kwargs) -> Tensor:
                    _transformation = dimensions

                    _inverse_transformation = inverse_transformation

                    if "transformation" in kwargs:
                        _transformation = kwargs["transformation"]

                        _inverse_transformation = _inverse_transform(_transformation)

                    if "updated_transformation" in kwargs:
                        _transformation = kwargs["updated_transformation"]

                    return u(input, apply_transform(_inverse_transformation, other))

                return displacement_fn, shift_fn

            def shift_fn(input: Tensor, other: Tensor, **kwargs) -> Tensor:
                _transformation = dimensions

                _inverse_transformation = inverse_transformation

                if "transformation" in kwargs:
                    _transformation = kwargs["transformation"]

                    _inverse_transformation = _inverse_transform(
                        _transformation,
                    )

                if "updated_transformation" in kwargs:
                    _transformation = kwargs["updated_transformation"]

                return input + apply_transform(_inverse_transformation, other)

            return displacement_fn, shift_fn

        def displacement_fn(
            input: Tensor,
            other: Tensor,
            *,
            perturbation: Tensor | None = None,
            **kwargs,
        ) -> Tensor:
            _transformation = dimensions

            _inverse_transformation = inverse_transformation

            if "transformation" in kwargs:
                _transformation = kwargs["transformation"]

                _inverse_transformation = _inverse_transform(_transformation)

            if "updated_transformation" in kwargs:
                _transformation = kwargs["updated_transformation"]

            input = apply_transform(_inverse_transformation, input)
            other = apply_transform(_inverse_transformation, other)

            if len(input.shape) != 1:
                raise ValueError

            if input.shape != other.shape:
                raise ValueError

            displacement = apply_transform(
                _transformation,
                torch.remainder(input - other + 1.0 * 0.5, 1.0) - 1.0 * 0.5,
            )

            if perturbation is not None:
                return _apply_transform(displacement, perturbation)

            return displacement

        if remapped:

            def u(input: Tensor, other: Tensor) -> Tensor:
                return torch.remainder(input + other, 1.0)

            def shift_fn(input: Tensor, other: Tensor, **kwargs) -> Tensor:
                _transformation = dimensions

                _inverse_transformation = inverse_transformation

                if "transformation" in kwargs:
                    _transformation = kwargs["transformation"]

                    _inverse_transformation = _inverse_transform(
                        _transformation,
                    )

                if "updated_transformation" in kwargs:
                    _transformation = kwargs["updated_transformation"]

                return apply_transform(
                    _transformation,
                    u(
                        apply_transform(_inverse_transformation, input),
                        apply_transform(_inverse_transformation, other),
                    ),
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

        displacement = torch.remainder(input - other + dimensions * 0.5, dimensions)

        if perturbation is not None:
            return _apply_transform(displacement - dimensions * 0.5, perturbation)

        return displacement - dimensions * 0.5

    if remapped:

        def shift_fn(input: Tensor, other: Tensor, **_) -> Tensor:
            return torch.remainder(input + other, dimensions)
    else:

        def shift_fn(input: Tensor, other: Tensor, **_) -> Tensor:
            return input + other

    return displacement_fn, shift_fn
