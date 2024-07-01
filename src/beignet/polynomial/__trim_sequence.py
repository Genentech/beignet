from torch import Tensor


def _trim_sequence(input: Tensor) -> Tensor:
    if input.shape[0] == 0:
        output = input
    else:
        index = 0

        for index in range(input.shape[0] - 1, -1, -1):
            if input[index] != 0:
                break

        output = input[: index + 1]

    return output
