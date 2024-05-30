import torch
from torch import Tensor


def _normalize_cell_size(box: Tensor, cutoff: float) -> Tensor:
    """
    Normalize the cell size given the bounding box dimensions and a cutoff value.

    Parameters
    ----------
    box : Tensor
        A tensor representing the dimensions of the bounding box. The shape of
        the tensor can be:
        - Scalar (0D): representing a uniform box.
        - 1D: a vector representing the side lengths of the box.
        - 2D: a matrix representing the side lengths and angles between them.

    cutoff : float
        The cutoff value used for normalization.

    Returns
    -------
    Tensor
        A tensor representing the normalized cell size.

    Raises
    ------
    ValueError
        If `box` has more than 3 dimensions or if the shape is not supported.
    """
    if box.ndim == 0:
        return cutoff / box

    if box.ndim == 1:
        return cutoff / torch.min(box)

    if box.ndim == 2:
        if box.shape[0] == 1:
            return 1 / torch.floor(box[0, 0] / cutoff)

        if box.shape[0] == 2:
            xx = box[0, 0]
            yy = box[1, 1]
            xy = box[0, 1] / yy

            nx = xx / torch.sqrt(1 + xy**2)
            ny = yy

            nmin = torch.floor(torch.min(torch.tensor([nx, ny])) / cutoff)

            return 1 / torch.where(nmin == 0, 1, nmin)

        if box.shape[0] == 3:
            xx = box[0, 0]
            yy = box[1, 1]
            zz = box[2, 2]
            xy = box[0, 1] / yy
            xz = box[0, 2] / zz
            yz = box[1, 2] / zz

            nx = xx / torch.sqrt(1 + xy**2 + (xy * yz - xz) ** 2)
            ny = yy / torch.sqrt(1 + yz**2)
            nz = zz

            nmin = torch.floor(torch.min(torch.tensor([nx, ny, nz])) / cutoff)
            return 1 / torch.where(nmin == 0, 1, nmin)
        else:
            raise ValueError
    else:
        raise ValueError
