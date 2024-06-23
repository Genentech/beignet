import torch

from .__as_series import _as_series


def physicists_hermite_series_companion(input):
    (input,) = _as_series([input])

    if len(input) < 2:
        raise ValueError

    if len(input) == 2:
        return torch.tensor(
            [[-0.5 * input[0] / input[1]]],
            dtype=input.dtype,
        )

    n = len(input) - 1

    output = torch.zeros([n, n], dtype=input.dtype)

    scl = torch.concatenate(
        (
            torch.tensor([1.0], dtype=input.dtype),
            1.0 / torch.sqrt(2.0 * torch.arange(n - 1, 0, -1, dtype=input.dtype)),
        )
    )
    scl = torch.cumprod(scl.flip(dims=(0,)), dim=0)

    mat_flat = output.view(-1)
    top_indices = torch.arange(1, len(mat_flat), n + 1)
    bot_indices = torch.arange(n, len(mat_flat), n + 1)

    mat_flat[top_indices] = torch.sqrt(0.5 * torch.arange(1, n, dtype=input.dtype))
    mat_flat[bot_indices] = mat_flat[top_indices]

    output = output.view(n, n)

    output[:, -1] = output[:, -1] - (scl * input[:-1] / (2.0 * input[-1]))

    return output
