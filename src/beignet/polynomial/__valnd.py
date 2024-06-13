import numpy


def _valnd(val_f, c, *args):
    args = [numpy.asanyarray(a) for a in args]
    shape0 = args[0].shape
    if not all((a.shape == shape0 for a in args[1:])):
        if len(args) == 3:
            raise ValueError("x, y, z are incompatible")
        elif len(args) == 2:
            raise ValueError("x, y are incompatible")
        else:
            raise ValueError("ordinates are incompatible")
    it = iter(args)
    x0 = next(it)

    c = val_f(x0, c)
    for xi in it:
        c = val_f(xi, c, tensor=False)
    return c
