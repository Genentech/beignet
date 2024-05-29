import beignet
import torch


def test_find_root_chandrupatla():
    c = torch.linspace(2, 100, 1001, dtype=torch.float64)

    def f(x):
        return x.pow(2) - c

    # we don't want to put the root in exactly the center of the interval
    a = c.sqrt() - 1.1
    b = c.sqrt() + 1.0

    x, meta = beignet.root(f, a, b)

    assert meta["converged"].all()

    torch.testing.assert_close(x, c.sqrt(), atol=1e-12, rtol=5e-11)
