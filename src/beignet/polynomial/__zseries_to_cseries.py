def _zseries_to_cseries(zs):
    n = (zs.size + 1) // 2
    c = zs[n - 1 :].copy()
    c[1:n] *= 2
    return c
