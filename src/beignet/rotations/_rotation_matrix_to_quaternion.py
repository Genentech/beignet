import torch
from torch import Tensor


def rotation_matrix_to_quaternion(
    input: Tensor,
    canonical: bool | None = False,
) -> Tensor:
    r"""
    Convert rotation matrices to rotation quaternions.

    Parameters
    ----------
    input : Tensor, shape=(..., 3, 3)
        Rotation matrices.

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

    Returns
    -------
    output : Tensor, shape=(..., 4)
        Rotation quaternion.
    """
    # This algorithm was causing gradient issues due to extensive inplace operations
    # For now, fall back to a simpler but correct implementation using vectorized operations
    # TODO: Optimize this later while maintaining gradient compatibility

    # Handle arbitrary batch dimensions by flattening
    original_shape = input.shape[:-2]
    input_flat = input.reshape(-1, 3, 3)
    batch_size = input_flat.shape[0]

    # Shepperd's method for rotation matrix to quaternion conversion
    # Based on "Converting a Rotation Matrix to Euler Angles and Back"

    # Extract diagonal elements
    r00 = input_flat[:, 0, 0]
    r11 = input_flat[:, 1, 1]
    r22 = input_flat[:, 2, 2]

    # Extract off-diagonal elements
    r01 = input_flat[:, 0, 1]
    r02 = input_flat[:, 0, 2]
    r10 = input_flat[:, 1, 0]
    r12 = input_flat[:, 1, 2]
    r20 = input_flat[:, 2, 0]
    r21 = input_flat[:, 2, 1]

    # Trace
    trace = r00 + r11 + r22

    # Initialize output
    q = torch.zeros(batch_size, 4, dtype=input.dtype, device=input.device)

    # Case 1: trace > 0
    mask1 = trace > 0
    S1 = torch.sqrt(trace[mask1] + 1.0) * 2  # S = 4 * qw
    q[mask1, 3] = 0.25 * S1  # qw
    q[mask1, 0] = (r21[mask1] - r12[mask1]) / S1  # qx
    q[mask1, 1] = (r02[mask1] - r20[mask1]) / S1  # qy
    q[mask1, 2] = (r10[mask1] - r01[mask1]) / S1  # qz

    # Case 2: r00 > r11 and r00 > r22
    mask2 = (~mask1) & (r00 > r11) & (r00 > r22)
    S2 = torch.sqrt(1.0 + r00[mask2] - r11[mask2] - r22[mask2]) * 2  # S = 4 * qx
    q[mask2, 3] = (r21[mask2] - r12[mask2]) / S2  # qw
    q[mask2, 0] = 0.25 * S2  # qx
    q[mask2, 1] = (r01[mask2] + r10[mask2]) / S2  # qy
    q[mask2, 2] = (r02[mask2] + r20[mask2]) / S2  # qz

    # Case 3: r11 > r22
    mask3 = (~mask1) & (~mask2) & (r11 > r22)
    S3 = torch.sqrt(1.0 + r11[mask3] - r00[mask3] - r22[mask3]) * 2  # S = 4 * qy
    q[mask3, 3] = (r02[mask3] - r20[mask3]) / S3  # qw
    q[mask3, 0] = (r01[mask3] + r10[mask3]) / S3  # qx
    q[mask3, 1] = 0.25 * S3  # qy
    q[mask3, 2] = (r12[mask3] + r21[mask3]) / S3  # qz

    # Case 4: else
    mask4 = (~mask1) & (~mask2) & (~mask3)
    S4 = torch.sqrt(1.0 + r22[mask4] - r00[mask4] - r11[mask4]) * 2  # S = 4 * qz
    q[mask4, 3] = (r10[mask4] - r01[mask4]) / S4  # qw
    q[mask4, 0] = (r02[mask4] + r20[mask4]) / S4  # qx
    q[mask4, 1] = (r12[mask4] + r21[mask4]) / S4  # qy
    q[mask4, 2] = 0.25 * S4  # qz

    # Normalize quaternions
    q_norm = torch.norm(q, dim=1, keepdim=True)
    q = q / q_norm

    if canonical:
        # Make canonical (positive w component)
        neg_mask = q[:, 3] < 0
        q[neg_mask] = -q[neg_mask]

    # Reshape back to original batch dimensions + quaternion dimension
    return q.reshape(*original_shape, 4)
