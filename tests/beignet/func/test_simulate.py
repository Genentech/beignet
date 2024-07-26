import pytest
import torch

from beignet.func import space
from beignet.func._quantity import kinetic_energy, temperature
from beignet.func._simulate import ensemble

PARTICLE_COUNT = 1000
DYNAMICS_STEPS = 800
SHORT_DYNAMICS_STEPS = 20
STOCHASTIC_SAMPLES = 5
SPATIAL_DIMENSION = [2, 3]
COORDS = ['fractional', 'real']

LANGEVIN_PARTICLE_COUNT = 8000
LANGEVIN_DYNAMICS_STEPS = 8000

BROWNIAN_PARTICLE_COUNT = 8000
BROWNIAN_DYNAMICS_STEPS = 8000

POSITION_DTYPE = [torch.float32, torch.float64]


test_cases = [
    {
        "dtype": dtype,
        "dim": dim,
    }
    for dtype in POSITION_DTYPE
    for dim in SPATIAL_DIMENSION
]

params = [(case["dtype"], case["dim"]) for case in test_cases]


ke_fn = lambda p, m: kinetic_energy(momentum=p, mass=m)
kT_fn = lambda p, m: temperature(momentum=p, mass=m)


@pytest.mark.parametrize("dtype, dim", params)
def test_nve_ensemble(dim, dtype):
    torch.manual_seed(0)
    R = torch.randn((PARTICLE_COUNT, dim), dtype=dtype)
    R0 = torch.randn((PARTICLE_COUNT, dim), dtype=dtype)
    mass = (5.0 - 0.1) * torch.rand((PARTICLE_COUNT,), dtype=dtype) + 0.1

    _, shift = space(box=None)

    E = lambda R, **kwargs: torch.sum((R - R0) ** 2)

    init_fn, apply_fn = ensemble(E, shift, 1e-3, kind="NVE")

    state = init_fn(R, kT=0.5, mass=mass)

    E_T = lambda state: E(state.position) + ke_fn(state.momentum, state.mass)
    E_initial = E_T(state)

    state = apply_fn(state)
    E_total = E_T(state)
    torch.assert_allclose(E_total, E_initial, rtol=0.01)
    assert state.position.dtype == dtype
