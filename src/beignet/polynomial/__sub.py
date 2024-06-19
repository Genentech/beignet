from .__as_series import _as_series


def _sub(input, other):
    [input, other] = _as_series([input, other])

    if len(input) > len(other):
        input[: other.size] = input[: other.size] - other

        output = input
    else:
        other = -other

        other[: input.size] = other[: input.size] + input

        output = other

    if len(output) != 0 and output[-1] == 0:
        for index in range(len(output) - 1, -1, -1):
            if output[index] != 0:
                break

        output = output[: index + 1]

    return output
