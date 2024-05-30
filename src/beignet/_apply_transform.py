import torch
from torch import Tensor
from torch.autograd import Function


# Only import torch._dynamo when necessary https://github.com/pytorch/pytorch/issues/110549
def conditional_import_torch_dynamo():
    import torch._dynamo

    return torch._dynamo


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
    if transform.ndim == 0:
        return input * transform

    indices = [chr(ord("a") + index) for index in range(input.ndim - 1)]

    indices = "".join(indices)

    if transform.ndim == 1:
        return torch.einsum(
            "i,...i->...i",
            transform,
            input,
        )

    if transform.ndim == 2:
        return torch.einsum(
            f"ij,...j->...i",
            transform,
            input,
        )

    raise ValueError


class _ApplyTransform(Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(transform: Tensor, position: Tensor) -> Tensor:
        """
        Return affine transformed position.

        Parameters
        ----------
        transform : Tensor
            Affine transformation matrix, must have shape
            `(dimension, dimension)`.

        position : Tensor
            Position, must have shape `(..., dimension)`.

        Returns
        -------
        Tensor
            Affine transformed position of shape `(..., dimension)`.
        """
        return _apply_transform(position, transform)

    @staticmethod
    def setup_context(ctx, inputs, output):
        transformation, position = inputs

        ctx.save_for_backward(transformation, position, output)

    @staticmethod
    def jvp(ctx, grad_transform: Tensor, grad_position: Tensor) -> (Tensor, Tensor):
        transformation, position, _ = ctx.saved_tensors

        _dynamo = conditional_import_torch_dynamo()

        output = _apply_transform(position, transformation)

        grad_output = grad_position + _apply_transform(position, grad_transform)

        return output, grad_output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> (Tensor, Tensor):
        _, _, output = ctx.saved_tensors

        _dynamo = conditional_import_torch_dynamo()

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
