from torch import Tensor

from beignet.operators import apply_rotation_matrix


def apply_transform(
    input: Tensor,
    transform: (Tensor, Tensor),
    inverse: bool | None = False,
) -> Tensor:
    r"""
    Applies three-dimensional transforms to vectors.

    Note
    ----
    This function interprets the rotation of the original frame to the final
    frame as either a projection, where it maps the components of vectors from
    the final frame to the original frame, or as a physical rotation,
    integrating the vectors into the original frame during the rotation
    process. Consequently, the vector components are maintained in the original
    frame’s perspective both before and after the rotation.

    Parameters
    ----------
    input : Tensor
        Each vector represents a vector in three-dimensional space. The number
        of rotation matrices, number of translation vectors, and number of
        input vectors must follow standard broadcasting rules: either one of
        them equals unity or they both equal each other.

    transform : (Tensor, Tensor)
        Transforms represented as a pair of rotation matrices and translation
        vectors. The rotation matrices must have the shape $(\ldots, 3, 3)$ and
        the translations must have the shape $(\ldots, 3)$.

    inverse : bool, optional
        If `True`, applies the inverse transformation (i.e., inverse rotation
        and negated translation) to the input vectors. Default, `False`.

    Returns
    -------
    output : Tensor
        Rotated and translated vectors.
    """
    rotation, translation = transform

    output = apply_rotation_matrix(input, rotation, inverse=inverse)

    if inverse:
        output = output - translation
    else:
        output = output + translation

    return output
