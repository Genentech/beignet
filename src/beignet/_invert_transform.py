import torch
from torch import Tensor

import beignet.operators


def invert_transform(input: (Tensor, Tensor)) -> (Tensor, Tensor):
    r"""
    Invert transforms.

    Parameters
    ----------
    input : (Tensor, Tensor)
        Transforms represented as a pair of rotation matrices and translation
        vectors. The rotation matrices must have the shape $(\ldots, 3, 3)$ and
        the translations must have the shape $(\ldots, 3)$.

    Returns
    -------
    output : (Tensor, Tensor)
        Inverted transforms represented as a pair of rotation matrices and
        translation vectors. The rotation matrices have the shape
        $(\ldots, 3, 3)$ and the translations have the shape $(\ldots, 3)$.
    """
    rotation, translation = input

    rotation = beignet.operators.invert_rotation_matrix(rotation)

    return rotation, -rotation @ torch.squeeze(
        torch.unsqueeze(translation, dim=-1), dim=-1
    )
