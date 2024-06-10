import torch
from torch import Tensor
from torch.autograd import Function


def _apply_transform(input: Tensor, transform: Tensor) -> Tensor:
    """
    Applies an affine transformation to the position vector.

    Parameters
    ----------
    input : Tensor
        Position, must have the shape `(..., dimension)`.

    transform : Tensor
        The affine transformation matrix, must be a scalar, a vector, or a
        matrix with the shape `(dimension, dimension)`.

    Returns
    -------
    Tensor
        Affine transformed position vector, has the same shape as the
        position vector.
    """
    match transform.ndim:
        case 0:
            return input * transform
        case 1:
            return torch.einsum("i,...i->...i", transform, input)
        case 2:
            return torch.einsum("ij,...j->...i", transform, input)
        case _:
            raise ValueError


class _ApplyTransform(Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(transform: Tensor, input: Tensor) -> Tensor:
        """
        Return affine transformed position.

        Parameters
        ----------
        transform : Tensor
            Affine transformation matrix, must have shape
            `(dimension, dimension)`.

        input : Tensor
            Position, must have shape `(..., dimension)`.

        Returns
        -------
        Tensor
            Affine transformed position of shape `(..., dimension)`.
        """
        return _apply_transform(input, transform)

    @staticmethod
    def setup_context(ctx, inputs, output):
        transform, input = inputs

        ctx.save_for_backward(transform, input, output)

    @staticmethod
    def jvp(ctx, grad_transform: Tensor, grad_input: Tensor) -> (Tensor, Tensor):
        transform, input, _ = ctx.saved_tensors

        output = _apply_transform(input, transform)

        grad_output = grad_input + _apply_transform(input, grad_transform)

        return output, grad_output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> (Tensor, Tensor):
        _, _, output = ctx.saved_tensors

        return output, grad_output


def apply_transform(input: Tensor, transform: Tensor) -> Tensor:
    """
    Return affine transformed position.

    Parameters
    ----------
    input : Tensor
        Position, must have shape `(..., dimension)`.

    transform : Tensor
        Affine transformation matrix, must have shape
        `(dimension, dimension)`.

    Returns
    -------
    Tensor
        Affine transformed position of shape `(..., dimension)`.
    """
    return _ApplyTransform.apply(transform, input)
