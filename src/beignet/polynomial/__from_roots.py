import torch

from .__as_series import _as_series


def _from_roots(line_f, mul_f, roots):
    if len(roots) == 0:
        return torch.ones(1, dtype=roots.dtype)
    else:
        (roots,) = _as_series([roots], trim=False)

        roots = torch.sort(roots)

        p = []

        for root in roots:
            p.append(line_f(-root, 1))

        n = len(p)

        while n > 1:
            m, r = divmod(n, 2)

            tmp = [mul_f(p[i], p[i + m]) for i in range(m)]

            if r:
                tmp[0] = mul_f(tmp[0], p[-1])

            p = tmp

            n = m

        return p[0]
