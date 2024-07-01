# cheb  : chebyshev_series
# herm  : physicists_hermite_series
# herme : probabilists_hermite_series
# lag   : laguerre_series
# leg   : legendre_series
# poly  : power_series

import torch.linalg

from .__as_series import _as_series
from .__nonzero import _nonzero
from .__trim_coefficients import _trim_coefficients
from .__vandermonde import _vandermonde
from ._chebadd import chebadd
from ._chebcompanion import chebcompanion
from ._chebline import chebline
from ._chebmul import chebmul
from ._chebpts1 import chebpts1
from ._chebsub import chebsub
from ._chebval import chebval
from ._chebvander import chebvander
from ._hermadd import hermadd
from ._hermcompanion import hermcompanion
from ._hermeadd import hermeadd
from ._hermecompanion import hermecompanion
from ._hermeline import hermeline
from ._hermemul import hermemul
from ._hermemulx import hermemulx
from ._hermesub import hermesub
from ._hermeval import hermeval
from ._hermevander import hermevander
from ._hermline import hermline
from ._hermmul import hermmul
from ._hermmulx import hermmulx
from ._hermsub import hermsub
from ._hermval import hermval
from ._hermvander import hermvander
from ._lagadd import lagadd
from ._lagcompanion import lagcompanion
from ._lagline import lagline
from ._lagmul import lagmul
from ._lagmulx import lagmulx
from ._lagsub import lagsub
from ._lagval import lagval
from ._lagvander import lagvander
from ._legadd import legadd
from ._legcompanion import legcompanion
from ._legline import legline
from ._legmul import legmul
from ._legmulx import legmulx
from ._legsub import legsub
from ._legval import legval
from ._legvander import legvander
from ._polyadd import polyadd
from ._polycompanion import polycompanion
from ._polyline import polyline
from ._polymul import polymul
from ._polymulx import polymulx
from ._polysub import polysub
from ._polyval import polyval
from ._polyvander import polyvander

torch.set_default_dtype(torch.float64)

chebdomain = torch.tensor([-1.0, 1.0])
chebone = torch.tensor([1.0])
chebx = torch.tensor([0.0, 1.0])
chebzero = torch.tensor([0.0])
hermdomain = torch.tensor([-1.0, 1.0])
hermedomain = torch.tensor([-1.0, 1.0])
hermeone = torch.tensor([1.0])
hermex = torch.tensor([0.0, 1.0])
hermezero = torch.tensor([0.0])
hermone = torch.tensor([1.0])
hermx = torch.tensor([0.0, 1.0 / 2.0])
hermzero = torch.tensor([0.0])
lagdomain = torch.tensor([0.0, 1.0])
lagone = torch.tensor([1.0])
lagx = torch.tensor([1.0, -1.0])
lagzero = torch.tensor([0.0])
legdomain = torch.tensor([-1.0, 1.0])
legone = torch.tensor([1.0])
legx = torch.tensor([0.0, 1.0])
legzero = torch.tensor([0.0])
polydomain = torch.tensor([-1.0, 1.0])
polyone = torch.tensor([1.0])
polyx = torch.tensor([0.0, 1.0])
polyzero = torch.tensor([0.0])

chebtrim = _trim_coefficients

hermetrim = _trim_coefficients

hermtrim = _trim_coefficients

lagtrim = _trim_coefficients

legtrim = _trim_coefficients

polytrim = _trim_coefficients

__all__ = [
    "chebdomain",
    "chebone",
    "chebtrim",
    "chebx",
    "chebzero",
    "hermdomain",
    "hermedomain",
    "hermeone",
    "hermetrim",
    "hermex",
    "hermezero",
    "hermone",
    "hermtrim",
    "hermx",
    "hermzero",
    "lagdomain",
    "lagone",
    "lagtrim",
    "lagx",
    "lagzero",
    "legdomain",
    "legone",
    "legtrim",
    "legx",
    "legzero",
    "polydomain",
    "polyone",
    "polytrim",
    "polyx",
    "polyzero",
]
