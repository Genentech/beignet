from .__cseries_to_zseries import _cseries_to_zseries
from .__zseries_mul import _zseries_mul
from .__zseries_to_cseries import _zseries_to_cseries
from ._as_series import as_series
from ._trimseq import trimseq


def chebmul(c1, c2):
    [c1, c2] = as_series([c1, c2])

    z1 = _cseries_to_zseries(c1)

    z2 = _cseries_to_zseries(c2)

    prd = _zseries_mul(z1, z2)

    ret = _zseries_to_cseries(prd)

    return trimseq(ret)
