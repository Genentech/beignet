def _trim_sequence(x):
    if len(x) == 0 or x[-1] != 0:
        output = x
    else:
        for index in range(len(x) - 1, -1, -1):
            if x[index] != 0:
                break

        output = x[: index + 1]

    return output
