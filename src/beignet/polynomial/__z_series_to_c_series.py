def _z_series_to_c_series(input):
    n = (input.size + 1) // 2
    output = input[n - 1 :]
    output[1:n] = output[1:n] * 2
    return output
