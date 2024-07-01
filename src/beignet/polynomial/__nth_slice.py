def _nth_slice(
    i,
    ndim,
):
    sl = [None] * ndim
    sl[i] = slice(None)
    return tuple(sl)
