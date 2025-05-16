import pytest
import torch

from beignet.structure import Rigid


def test_broadcast_compose():
    rigid1 = Rigid.rand(3, 4)
    rigid2 = Rigid.rand(3, 1)

    rigid3 = rigid2.compose(rigid1)
    assert rigid3.shape == (3, 4)


def test_rigid_inverse():
    B = 100
    rigid = Rigid.rand(B)
    x = torch.rand(B, 3)

    torch.testing.assert_close(rigid.inverse().compose(rigid)(x), x)


def test_kabsch():
    B, N, D = (101, 20, 3)
    T = Rigid.rand(B, 1)
    x = torch.randn(B, N, D)
    y = T(x)
    T_kabsch = Rigid.kabsch(y, x)
    y_kabsch = T_kabsch(x)
    torch.testing.assert_close(T_kabsch.t, T.t)
    torch.testing.assert_close(T_kabsch.r, T.r)
    torch.testing.assert_close(y_kabsch, y)


def test_kabsch_masked():
    B, N, D = (101, 20, 3)

    T = Rigid.rand(B)
    x = torch.randn(B, N, D)
    y = T[:, None](x)

    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -1] = False
    y[~mask] = 0.0

    # we get the wrong answer if we don't mask out the modified coordinates
    T_kabsch = Rigid.kabsch(y, x)
    y_kabsch = T_kabsch(x)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(y[mask], y_kabsch[mask])

    # if we mask out the modified coordinates using indexing we get the right answer again
    T_kabsch = Rigid.kabsch(y[:, :-1, :], x[:, :-1, :])
    y_kabsch = T_kabsch(x)
    torch.testing.assert_close(y_kabsch[mask], y[mask])

    # instead of using index we can provide the mask directly through the weights parameter
    T_kabsch = Rigid.kabsch(y, x, weights=mask)
    y_kabsch = T_kabsch(x)
    torch.testing.assert_close(y_kabsch[mask], y[mask])


def test_kabsch_multiple_batch_dimensions():
    B1, B2, N, D = (1001, 7, 20, 3)
    T = Rigid.rand(B1, B2, 1)
    x = torch.randn(B1, B2, N, D)
    y = T(x)
    T_kabsch = Rigid.kabsch(y, x)
    y_kabsch = T_kabsch(x)
    torch.testing.assert_close(T_kabsch.t, T.t)
    torch.testing.assert_close(T_kabsch.r, T.r)
    torch.testing.assert_close(y_kabsch, y)
