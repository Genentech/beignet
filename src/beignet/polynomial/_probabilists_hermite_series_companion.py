import torch

from .__as_series import _as_series


def probabilists_hermite_series_companion(input):
    (input,) = _as_series([input])

    if len(input) < 2:
        raise ValueError("Input must have at least 2 elements")

    if len(input) == 2:
        return torch.tensor([[-input[0] / input[1]]], dtype=input.dtype)

    n = len(input) - 1

    output = torch.zeros((n, n), dtype=input.dtype)

    # Compute scl using torch operations
    scl = torch.cat(
        (
            torch.tensor([1.0], dtype=input.dtype),
            1.0 / torch.sqrt(torch.arange(n - 1, 0, -1, dtype=input.dtype)),
        )
    )
    scl = torch.cumprod(scl.flip(dims=(0,)), dim=0)

    # Equivalent of top and bot assignments
    output = output.view(-1)
    top_indices = torch.arange(1, len(output), n + 1)
    bot_indices = torch.arange(n, len(output), n + 1)

    output[top_indices] = torch.sqrt(torch.arange(1, n, dtype=input.dtype))
    output[bot_indices] = output[top_indices]

    output = output.view(n, n)

    output[:, -1] -= scl * input[:-1] / input[-1]

    return output
