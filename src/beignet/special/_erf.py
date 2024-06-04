from ._erfc import erfc


def erf(z):
    return 1 - erfc(z)
