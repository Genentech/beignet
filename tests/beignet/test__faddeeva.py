import scipy
import torch
from beignet import erf


def test_erf():
    # TODO test if we match scipy behavior when results overflows
    _x = torch.linspace(-10.0, 10.0, 101, dtype=torch.float64)
    _y = torch.linspace(-10.0, 10.0, 51, dtype=torch.float64)

    x, y = torch.meshgrid(_x, _y, indexing="xy")
    z = x + 1j * y
    val = erf(z)
    ref = scipy.special.erf(z)

    torch.testing.assert_close(ref, val, rtol=1e-12, atol=5e-11)
