from torch import Tensor


def invert_quaternion(
    input: Tensor,
    canonical: bool = False,
) -> Tensor:
    r"""
    Invert rotation quaternions.

    Parameters
    ----------
    input : Tensor, shape (..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

    Returns
    -------
    inverted_quaternions : Tensor, shape (..., 4)
        Inverted rotation quaternions.
    """
    input[:, :3] = -input[:, :3]

    return input
