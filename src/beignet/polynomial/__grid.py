def _grid(func, input, *xs):
    for x in xs:
        input = func(x, input)

    return input
