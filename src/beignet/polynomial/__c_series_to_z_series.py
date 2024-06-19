import torch


def _c_series_to_z_series(input):
    n = input.shape[-1]
    output = torch.zeros(2 * n - 1, dtype=input.dtype)
    output[n - 1 :] = input / 2
    return output + torch.flip(output, dims=[0])
