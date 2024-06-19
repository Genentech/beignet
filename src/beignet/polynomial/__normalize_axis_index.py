def _normalize_axis_index(axis, ndim):
    if axis < 0:
        axis = axis + ndim

    return axis
