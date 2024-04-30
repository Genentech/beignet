import torch
import torch.nn.functional
from torch import Tensor


def per_residue_local_distance_difference_test(input: Tensor) -> Tensor:
    """
    Parameters
    ----------
    input : Tensor

    Returns
    -------
    output : Tensor
    """
    probs = torch.nn.functional.softmax(input, dim=-1)

    bins = input.shape[-1]

    step = 1.0 / bins

    bounds = torch.arange(0.5 * step, 1.0, step)

    indexes = (1,) * len(probs.shape[:-1])
    output = bounds.view(*indexes, *bounds.shape)
    output = probs * output
    output = torch.sum(output, dim=-1)

    return output * 100
