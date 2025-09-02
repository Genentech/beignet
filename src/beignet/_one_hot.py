import torch
from torch import Tensor


def one_hot(x: Tensor, v_bins: Tensor) -> Tensor:
    r"""
    One-hot encoding with nearest bin assignment.

    This function creates a one-hot encoding by finding the nearest bin
    for each value in x and setting the corresponding bin to 1.

    Parameters
    ----------
    x : Tensor
        Input values to encode
    v_bins : Tensor
        Bin centers for one-hot encoding

    Returns
    -------
    p : Tensor
        One-hot encoded tensor with shape (..., len(v_bins))

    Examples
    --------
    >>> import torch
    >>> from beignet import one_hot
    >>> x = torch.tensor([1.2, 3.7, 5.1])
    >>> bins = torch.tensor([1.0, 3.0, 5.0, 7.0])
    >>> result = one_hot(x, bins)
    >>> result.shape
    torch.Size([3, 4])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 4: One-hot encoding with nearest bin
    """
    x_flat = x.reshape(-1)

    output = torch.zeros(
        x_flat.shape[0],
        v_bins.shape[0],
        dtype=x.dtype,
        device=x.device,
    )

    difference = torch.unsqueeze(x_flat, -1) - torch.unsqueeze(v_bins, 0)
    difference = torch.abs(difference)  # (n_vals, n_bins)

    index = torch.argmin(difference, dim=-1)

    output[torch.arange(x_flat.shape[0]), index] = 1

    return torch.reshape(output, [*x.shape, v_bins.shape[0]])
