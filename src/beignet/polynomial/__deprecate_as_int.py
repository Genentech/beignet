import operator
import warnings


def _deprecate_as_int(x, desc):
    try:
        return operator.index(x)
    except TypeError as e:
        try:
            ix = int(x)
        except TypeError:
            pass
        else:
            if ix == x:
                warnings.warn(
                    f"In future, this will raise TypeError, as {desc} will "
                    "need to be an integer not just an integral float.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                return ix

        raise TypeError(f"{desc} must be an integer") from e
