from ._as_series import as_series
from ._trimseq import trimseq


def _add(c1, c2):
    [c1, c2] = as_series([c1, c2])
    if len(c1) > len(c2):
        c1[: c2.size] += c2
        ret = c1
    else:
        c2[: c1.size] += c1
        ret = c2
    return trimseq(ret)
