import torch
import torch.nn as nn
from torch import Tensor


class CentreRandomAugmentation(nn.Module):
    r"""
    Centre Random Augmentation for AlphaFold 3.

    This module implements Algorithm 19 exactly, applying random rotation and
    translation to center and augment atomic positions.

    Parameters
    ----------
    s_trans : float, default=1.0
        Translation scale in Ångstroms

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import CentreRandomAugmentation
    >>> batch_size, n_atoms = 2, 1000
    >>> module = CentreRandomAugmentation()
    >>>
    >>> x_t = torch.randn(batch_size, n_atoms, 3)
    >>> x_t_augmented = module(x_t)
    >>> x_t_augmented.shape
    torch.Size([2, 1000, 3])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 19: CentreRandomAugmentation
    """

    def __init__(self, s_trans: float = 1.0):
        super().__init__()

        self.s_trans = s_trans

    def uniform_random_rotation(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """
        Generate uniform random rotation matrices.

        Parameters
        ----------
        batch_size : int
            Number of rotation matrices to generate
        device : torch.device
            Device to create tensors on

        Returns
        -------
        R : Tensor, shape=(batch_size, 3, 3)
            Random rotation matrices
        """
        # Generate random quaternions and normalize them for uniform distribution
        # This is a simplified implementation - for true uniformity, use more sophisticated methods
        q = torch.randn(batch_size, 4, device=device, dtype=dtype)
        q = q / torch.norm(q, dim=-1, keepdim=True)

        # Convert quaternions to rotation matrices
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Rotation matrix from quaternion
        R = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)

        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)

        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y * z - x * w)

        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        return R

    def forward(self, x_t: Tensor) -> Tensor:
        r"""
        Forward pass implementing Algorithm 19 exactly.

        Implements the following steps:
        1. x̃_t ← x̃_t - mean_l x̃_t
        2. r = UniformRandomRotation()
        3. t̃ ∼ s_trans · N(0̃, I_3)
        4. x̃_t ← r · x̃_t + t̃
        5. return {x̃_t}

        Parameters
        ----------
        x_t : Tensor, shape=(batch_size, n_atoms, 3)
            Input atomic positions

        Returns
        -------
        x_t : Tensor, shape=(batch_size, n_atoms, 3)
            Augmented atomic positions
        """
        batch_size, n_atoms, _ = x_t.shape
        # Step 1: x̃_t ← x̃_t - mean_l x̃_t
        # Center the coordinates by subtracting the mean
        x_t = x_t - torch.mean(x_t, dim=1, keepdim=True)

        # Step 2: r = UniformRandomRotation()
        r = self.uniform_random_rotation(batch_size, x_t.device, x_t.dtype)

        # Step 3: t̃ ∼ s_trans · N(0̃, I_3)
        t = self.s_trans * torch.randn(
            batch_size,
            1,
            3,
            device=x_t.device,
            dtype=x_t.dtype,
        )

        # Step 4: x̃_t ← r · x̃_t + t̃
        # Apply rotation: (batch_size, 3, 3) @ (batch_size, n_atoms, 3) -> (batch_size, n_atoms, 3)
        # x_t @ r^T
        x_t = torch.bmm(x_t, r.transpose(-2, -1)) + t

        # Step 5: return {x̃_t}
        return x_t
