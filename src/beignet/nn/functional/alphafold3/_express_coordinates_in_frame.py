import torch
from torch import Tensor


def express_coordinates_in_frame(
    coordinates: Tensor,
    frame: Tensor,
) -> Tensor:
    r"""
    Express 3D coordinates in a local frame coordinate system.

    This implements Algorithm 29 from AlphaFold 3, which transforms 3D coordinates
    from global space into a local coordinate system defined by three frame atoms.
    The algorithm builds an orthonormal basis from the frame atoms and projects
    the coordinates onto this basis.

    Parameters
    ----------
    coordinates : Tensor, shape=(..., 3)
        3D coordinates to be expressed in the frame coordinate system.

    frame : Tensor, shape=(..., 3, 3)
        Frame definition as three 3D points [a, b, c] where:
        - a, b, c are the three atoms defining the local frame
        - Shape is (..., 3, 3) where frame[..., i, :] is the i-th atom coordinates

    Returns
    -------
    transformed : Tensor, shape=(..., 3)
        Coordinates expressed in the local frame coordinate system.

    Notes
    -----
    The algorithm follows these steps:

    1. Extract frame atoms: (a, b, c) = frame
    2. Compute normalized vectors: w₁ = (a - b) / ||a - b||, w₂ = (c - b) / ||c - b||
    3. Build orthonormal basis:
       - e₁ = (w₁ + w₂) / ||w₁ + w₂||
       - e₂ = (w₂ - w₁) / ||w₂ - w₁||
       - e₃ = e₁ × e₂
    4. Project coordinates: d = coordinates - b
    5. Transform: result = [d·e₁, d·e₂, d·e₃]

    This creates a local coordinate system centered at atom b, with axes defined
    by the geometric relationship between the three frame atoms.

    Examples
    --------
    >>> import torch
    >>> import beignet
    >>> batch_size = 2
    >>> # Define frame atoms (a, b, c)
    >>> frame = torch.randn(batch_size, 3, 3)
    >>> # Define coordinates to transform
    >>> coords = torch.randn(batch_size, 3)
    >>> # Express coordinates in frame
    >>> transformed = beignet.express_coordinates_in_frame(coords, frame)
    >>> transformed.shape
    torch.Size([2, 3])

    References
    ----------
    .. [1] AlphaFold 3 paper, Algorithm 29: Express coordinates in frame
    """
    # Validate input shapes
    if coordinates.shape[-1] != 3:
        raise ValueError(f"Coordinates must be 3D: got shape {coordinates.shape}")

    if frame.shape[-2:] != (3, 3):
        raise ValueError(f"Frame must have shape (..., 3, 3): got shape {frame.shape}")

    if coordinates.shape[:-1] != frame.shape[:-2]:
        raise ValueError(
            f"Batch dimensions must match: coordinates {coordinates.shape[:-1]} "
            f"vs frame {frame.shape[:-2]}"
        )

    eps = torch.finfo(coordinates.dtype).eps

    # Extract frame atoms (step 1)
    a = frame[..., 0, :]  # (..., 3)
    b = frame[..., 1, :]  # (..., 3)
    c = frame[..., 2, :]  # (..., 3)

    # Compute normalized direction vectors (steps 2-3)
    w1_unnorm = a - b  # (..., 3)
    w2_unnorm = c - b  # (..., 3)

    w1_norm = torch.norm(w1_unnorm, dim=-1, keepdim=True)  # (..., 1)
    w2_norm = torch.norm(w2_unnorm, dim=-1, keepdim=True)  # (..., 1)

    w1 = w1_unnorm / (w1_norm + eps)  # (..., 3)
    w2 = w2_unnorm / (w2_norm + eps)  # (..., 3)

    # Build orthonormal basis (steps 4-6)
    e1_unnorm = w1 + w2  # (..., 3)
    e1_norm = torch.norm(e1_unnorm, dim=-1, keepdim=True)  # (..., 1)
    e1 = e1_unnorm / (e1_norm + eps)  # (..., 3)

    e2_unnorm = w2 - w1  # (..., 3)
    e2_norm = torch.norm(e2_unnorm, dim=-1, keepdim=True)  # (..., 1)
    e2 = e2_unnorm / (e2_norm + eps)  # (..., 3)

    # Cross product for e3 = e1 × e2
    e3 = torch.cross(e1, e2, dim=-1)  # (..., 3)

    # Project onto frame basis (steps 7-8)
    d = coordinates - b  # (..., 3)

    # Compute dot products: d · e1, d · e2, d · e3
    proj1 = torch.sum(d * e1, dim=-1, keepdim=True)  # (..., 1)
    proj2 = torch.sum(d * e2, dim=-1, keepdim=True)  # (..., 1)
    proj3 = torch.sum(d * e3, dim=-1, keepdim=True)  # (..., 1)

    # Concatenate projections (step 8)
    transformed = torch.cat([proj1, proj2, proj3], dim=-1)  # (..., 3)

    return transformed
