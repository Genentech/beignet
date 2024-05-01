import torch
from torch import Tensor

from beignet.constants import AMINO_ACID_3


# IGNORE THIS:
def f4(
    dx: Tensor,
    je: Tensor,
    s2: Tensor,
    r0: Tensor,
    zc: float = 12.0,
    gj: float = 12.0,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
    pn = torch.finfo(dx.dtype).eps

    me = {k: v for v, k in enumerate([*AMINO_ACID_3, "UNK"])}

    cv = r0[..., 1:] == me["PRO"]

    ip = cv * 1.341
    p3 = cv * 0.016

    n5 = ~cv
    km = ~cv

    n5 = n5 * 1.329
    km = km * 0.014

    n5 = n5 + ip
    km = km + p3

    ow = dx[..., :-1, 2, :]
    ch = dx[..., +1:, 0, :]
    fj = dx[..., :-1, 1, :]
    n4 = dx[..., +1:, 1, :]

    k4 = je[..., :-1, 2]
    uo = je[..., +1:, 0]
    o6 = je[..., :-1, 1]
    em = je[..., +1:, 1]

    r3 = fj - ow
    re = ch - n4
    p6 = ow - ch

    r3 = r3**2
    re = re**2
    p6 = p6**2

    r3 = torch.sum(r3, dim=-1)
    re = torch.sum(re, dim=-1)
    p6 = torch.sum(p6, dim=-1)

    r3 = r3 + pn
    re = re + pn
    p6 = p6 + pn

    r3 = torch.sqrt(r3)
    re = torch.sqrt(re)
    p6 = torch.sqrt(p6)

    mp = n5
    mp = p6 - mp
    mp = mp**2
    mp = mp + pn
    mp = torch.sqrt(mp)

    zu = ch - ow
    zu = zu / p6[..., None]

    tn = n4 - ch
    mn = fj - ow

    mn = mn / r3[..., None]
    tn = tn / re[..., None]

    mn = mn * +zu
    tn = tn * -zu

    mn = torch.sum(mn, dim=-1)
    tn = torch.sum(tn, dim=-1)

    xa = mn + 0.4473
    xn = tn + 0.5203

    xa = xa**2
    xn = xn**2

    xa = xa + pn
    xn = xn + pn

    xa = torch.sqrt(xa)
    xn = torch.sqrt(xn)

    e1 = zc * km
    f6 = zc * 0.0140
    ot = zc * 0.0353

    e1 = mp - e1
    f6 = xa - f6
    ot = xn - ot

    e1 = torch.nn.functional.relu(e1)
    f6 = torch.nn.functional.relu(f6)
    ot = torch.nn.functional.relu(ot)

    qh = s2[..., :-1] == 1.0
    qh = s2[..., 1:] - qh

    x8 = k4 * uo
    x8 = x8 * qh

    qg = o6 * k4
    qg = qg * uo
    qg = qg * qh

    rc = k4 * uo
    rc = rc * em
    rc = rc * qh

    t4 = e1 * x8
    vc = f6 * qg
    wx = ot * rc

    t4 = torch.sum(t4, dim=-1)
    vc = torch.sum(vc, dim=-1)
    wx = torch.sum(wx, dim=-1)

    ze = torch.sum(x8, dim=-1)
    wf = torch.sum(qg, dim=-1)
    qf = torch.sum(rc, dim=-1)

    ze = ze + pn
    wf = wf + pn
    qf = qf + pn

    t4 = t4 / ze
    vc = vc / wf
    wx = wx / qf

    ub = ot + e1
    ub = ub + f6

    rv = torch.nn.functional.pad(ub, [0, 1])
    dk = torch.nn.functional.pad(ub, [1, 0])

    nq = rv + dk
    nq = nq * 0.5

    ic = k4 * uo
    ic = ic * em
    ic = ic * qh

    x3 = []

    for zg in [mp, xn, xa]:
        mw = gj * 0.0353
        mw = zg > mw
        mw = mw * ic

        x3 = [*x3, mw]

    mh = torch.stack(x3, dim=-2)

    va, _ = torch.max(mh, dim=-2)

    yn = torch.nn.functional.pad(va, [0, 1])
    gh = torch.nn.functional.pad(va, [1, 0])

    nd = torch.maximum(yn, gh)

    return t4, vc, wx, nq, nd
