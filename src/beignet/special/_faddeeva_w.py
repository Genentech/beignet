import torch


def _voigt_v_impl(x, y):
    # assumes x >= 0, y >= 0

    N = 11

    # h = math.sqrt(math.pi / (N + 1))
    h = 0.5116633539732443

    phi = (x / h) - (x / h).floor()

    k = torch.arange(N + 1, dtype=x.dtype, device=x.device)
    t = (k + 0.5) * h
    tau = k[1:] * h

    # equation 12
    w_m = (2 * h * y / torch.pi) * (
        torch.exp(-t.pow(2))
        * (t.pow(2) + x[..., None].pow(2) + y[..., None].pow(2))
        / (
            ((t - x[..., None]).pow(2) + y[..., None].pow(2))
            * ((t + x[..., None]).pow(2) + y[..., None].pow(2))
        )
    ).sum(dim=-1)

    # equation 13
    w_mm = (
        (
            2
            * torch.exp(-x.pow(2) + y.pow(2))
            * (
                torch.cos(2 * x * y)
                + torch.exp(2 * torch.pi * y / h)
                * torch.cos(2 * torch.pi * x / h - 2 * x * y)
            )
        )
        / (
            1
            + torch.exp(4 * torch.pi * y / h)
            + 2 * torch.exp(2 * torch.pi * y / h) * torch.cos(2 * torch.pi * x / h)
        )
    ) + w_m

    w_mt_1 = (
        2
        * torch.exp(-x.pow(2) + y.pow(2))
        * (
            torch.cos(2 * x * y)
            - torch.exp(2 * torch.pi * y / h)
            * torch.cos(2 * torch.pi * x / h - 2 * x * y)
        )
    ) / (
        1
        + torch.exp(4 * torch.pi * y / h)
        - 2 * torch.exp(2 * torch.pi * y / h) * torch.cos(2 * torch.pi * x / h)
    )

    w_mt_2 = (h * y) / (torch.pi * (x.pow(2) + y.pow(2)))

    w_mt_3 = (2 * h * y / torch.pi) * (
        torch.exp(-tau.pow(2))
        * (tau.pow(2) + x[..., None].pow(2) + y[..., None].pow(2))
        / (
            ((tau - x[..., None]).pow(2) + y[..., None].pow(2))
            * ((tau + x[..., None]).pow(2) + y[..., None].pow(2))
        )
    ).sum(dim=-1)

    # equation 14
    w_mt = w_mt_1 + w_mt_2 + w_mt_3

    return torch.where(
        y >= torch.maximum(x, torch.tensor(torch.pi / h)),
        w_m,
        torch.where((y < x) & (1 / 4 <= phi) & (phi <= 3 / 4), w_mt, w_mm),
    )


def _voigt_l_impl(x, y):
    # assumes x >= 0, y >= 0

    N = 11

    # h = math.sqrt(math.pi / (N + 1))
    h = 0.5116633539732443

    phi = (x / h) - (x / h).floor()

    k = torch.arange(N + 1, dtype=x.dtype, device=x.device)
    t = (k + 0.5) * h
    tau = k[1:] * h

    w_m = (2 * h * x / torch.pi) * (
        torch.exp(-t.pow(2))
        * (-t.pow(2) + x[..., None].pow(2) + y[..., None].pow(2))
        / (
            ((t - x[..., None]).pow(2) + y[..., None].pow(2))
            * ((t + x[..., None]).pow(2) + y[..., None].pow(2))
        )
    ).sum(dim=-1)

    # equation 13
    w_mm = (
        (
            -2
            * torch.exp(-x.pow(2) + y.pow(2))
            * (
                torch.sin(2 * x * y)
                - torch.exp(2 * torch.pi * y / h)
                * torch.sin(2 * torch.pi * x / h - 2 * x * y)
            )
        )
        / (
            1
            + torch.exp(4 * torch.pi * y / h)
            + 2 * torch.exp(2 * torch.pi * y / h) * torch.cos(2 * torch.pi * x / h)
        )
    ) + w_m

    w_mt_1 = (
        -2
        * torch.exp(-x.pow(2) + y.pow(2))
        * (
            torch.sin(2 * x * y)
            + torch.exp(2 * torch.pi * y / h)
            * torch.sin(2 * torch.pi * x / h - 2 * x * y)
        )
    ) / (
        1
        + torch.exp(4 * torch.pi * y / h)
        - 2 * torch.exp(2 * torch.pi * y / h) * torch.cos(2 * torch.pi * x / h)
    )

    w_mt_2 = (h * x) / (torch.pi * (x.pow(2) + y.pow(2)))

    w_mt_3 = (2 * h * x / torch.pi) * (
        torch.exp(-tau.pow(2))
        * (-tau.pow(2) + x[..., None].pow(2) + y[..., None].pow(2))
        / (
            ((tau - x[..., None]).pow(2) + y[..., None].pow(2))
            * ((tau + x[..., None]).pow(2) + y[..., None].pow(2))
        )
    ).sum(dim=-1)

    # equation 14
    w_mt = w_mt_1 + w_mt_2 + w_mt_3

    return torch.where(
        y >= torch.maximum(x, torch.tensor(torch.pi / h)),
        w_m,
        torch.where((y < x) & (1 / 4 <= phi) & (phi <= 3 / 4), w_mt, w_mm),
    )


def _faddeeva_w_impl(z):
    return _voigt_v_impl(z.real, z.imag) + 1j * _voigt_l_impl(z.real, z.imag)


# NOTE we use _voigt_v_impl and _voigt_l_impl to compute the real and complex parts
# separately instead of this because torch.compile does not support complex numbers
# def _faddeeva_w_impl_complex(z):
#    N = 11
#
#    # equation 11
#    # h = math.sqrt(math.pi / (N + 1))
#    h = 0.5116633539732443
#
#    x = z.real
#    y = z.imag
#
#    phi = (x / h) - (x / h).floor()
#
#    k = torch.arange(N + 1, dtype=x.dtype, device=z.device)
#    t = (k + 0.5) * h
#    tau = k[1:] * h
#
#    # equation 12
#    w_m = (2j * h * z / torch.pi) * (
#        torch.exp(-t.pow(2)) / (z[..., None].pow(2) - t.pow(2))
#    ).sum(dim=-1)
#
#    # equation 13
#    w_mm = (2 * torch.exp(-z.pow(2)) / (1 + torch.exp(-2j * torch.pi * z / h))) + w_m
#
#    # equation 14
#    w_mt = (
#        (2 * torch.exp(-z.pow(2)) / (1 - torch.exp(-2j * torch.pi * z / h)))
#        + (1j * h / (torch.pi * z))
#        + (2j * h * z / torch.pi)
#        * (torch.exp(-tau.pow(2)) / (z[..., None].pow(2) - tau.pow(2))).sum(dim=-1)
#    )
#
#    return torch.where(
#        y >= torch.maximum(x, torch.tensor(torch.pi / h)),
#        w_m,
#        torch.where((y < x) & (1 / 4 <= phi) & (phi <= 3 / 4), w_mt, w_mm),
#    )


def faddeeva_w(z: torch.Tensor):
    """Compute faddeeva w function using method described in [1].

    Parameterz
    ----------
    z: torch.Tensor
        complex input

    References
    ----------
    [1] Al Azah, Mohammad, and Simon N. Chandler-Wilde.
        "Computation of the complex error function using modified trapezoidal rules."
        SIAM Journal on Numerical Analysis 59.5 (2021): 2346-2367.
    """
    # use symmetries to map to upper right quadrant of complex plane
    imag_negative = z.imag < 0.0
    z = torch.where(z.imag < 0.0, -z, z)
    real_negative = z.real < 0.0
    z = torch.where(z.real < 0.0, -z.conj(), z)
    assert (z.real >= 0.0).all()
    assert (z.imag >= 0.0).all()

    out = _faddeeva_w_impl(z)
    out = torch.where(imag_negative, 2 * torch.exp(-z.pow(2)) - out, out)
    out = torch.where(real_negative, out.conj(), out)
    return out
