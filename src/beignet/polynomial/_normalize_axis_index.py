from numpy.exceptions import AxisError


def normalize_axis_index(axis, ndim):
    if axis < -ndim or axis >= ndim:
        msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
        raise AxisError(msg)

    if axis < 0:
        axis = axis + ndim
    return axis
