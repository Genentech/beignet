import numpy


def _evaluate(func, input, *xs):
    xs = [numpy.asanyarray(a) for a in xs]

    if not all((a.shape == xs[0].shape for a in xs[1:])):
        match len(xs):
            case 2:
                raise ValueError
            case 3:
                raise ValueError
            case _:
                raise ValueError

    xs = iter(xs)

    output = func(next(xs), input)

    for x in xs:
        output = func(x, output, tensor=False)

    return output
