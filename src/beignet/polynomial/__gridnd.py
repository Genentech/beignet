def _gridnd(val_f, c, *args):
    for xi in args:
        c = val_f(xi, c)
    return c
