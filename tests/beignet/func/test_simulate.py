import pytest
import torch

from beignet._lennard_jones_neighbor_list_potential import (
    lennard_jones_neighbor_list_potential,
)
from beignet._lennard_jones_pair_potential import lennard_jones_pair_potential
from beignet.func import space
from beignet.func._quantity import kinetic_energy, temperature
from beignet.func._simulate import ensemble

PARTICLE_COUNT = 1000
DYNAMICS_STEPS = 800
SHORT_DYNAMICS_STEPS = 20
STOCHASTIC_SAMPLES = 5
SPATIAL_DIMENSION = [2, 3]
COORDS = ["fractional", "real"]

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
nve_neighbor_list_params = [(torch.float64, 2), (torch.float64, 3)]


ke_fn = lambda p, m: kinetic_energy(momentum=p, mass=m)
kT_fn = lambda p, m: temperature(momentum=p, mass=m)


@pytest.mark.parametrize("dtype, dim", params)
def test_nve_ensemble(dim, dtype):
    torch.manual_seed(0)
    R = torch.randn((PARTICLE_COUNT, dim), dtype=dtype)
    R0 = torch.randn((PARTICLE_COUNT, dim), dtype=dtype)
    mass = (5.0 - 0.1) * torch.rand((PARTICLE_COUNT,), dtype=dtype) + 0.1

    _, shift = space(box=None)

    energy_fn = lambda R, **kwargs: torch.sum((R - R0) ** 2)

    setup_fn, step_fn = ensemble(energy_fn, shift, 1e-3, kind="NVE")

    state = setup_fn(R, temperature=0.5, masses=mass)

    E_T = lambda state: energy_fn(state.positions) + ke_fn(
        state.momentums, state.masses
    )
    E_initial = E_T(state)

    state = step_fn(state)
    E_total = E_T(state)
    assert torch.allclose(E_total, E_initial, rtol=0.01)
    assert state.positions.dtype == dtype


# TODO (isaacsoh) doesn't work for float32
@pytest.mark.parametrize("dtype, dim", nve_neighbor_list_params)
def test_nve_neighbor_list(dim, dtype):
    particles_per_side = 8
    spacing = 1.25

    tol = 5e-12 if dtype == torch.float64 else 5e-3

    box = torch.tensor(particles_per_side * spacing)
    if dim == 2:
        position = (
            torch.stack(
                [
                    torch.tensor([i, j], dtype=dtype)
                    for i in range(particles_per_side)
                    for j in range(particles_per_side)
                ]
            )
            * spacing
        )
    elif dim == 3:
        position = (
            torch.stack(
                [
                    torch.tensor([i, j, k], dtype=dtype)
                    for i in range(particles_per_side)
                    for j in range(particles_per_side)
                    for k in range(particles_per_side)
                ]
            )
            * spacing
        )

    displacement, shift = space(box=box)

    neighbor_fn, energy_fn = lennard_jones_neighbor_list_potential(displacement, box)
    exact_energy_fn = lennard_jones_pair_potential(displacement)

    setup_fn, step_fn = ensemble(energy_fn, shift, 1e-3, kind="NVE")
    exact_setup_fn, exact_step_fn = ensemble(exact_energy_fn, shift, 1e-3, kind="NVE")

    nbrs = neighbor_fn.setup_fn(position)
    state = setup_fn(position, temperature=0.5, neighbor_list=nbrs)
    exact_state = exact_setup_fn(position, temperature=0.5)

    def body_fn(state, nbrs, exact_state):
        nbrs = neighbor_fn.update_fn(state.positions, nbrs)
        state = step_fn(state, neighbor_list=nbrs)
        exact_state = exact_step_fn(exact_state)
        return state, nbrs, exact_state

    step = 0
    for i in range(2):
        for _ in range(2):
            state, nbrs, exact_state = body_fn(state, nbrs, exact_state)
        if nbrs.did_buffer_overflow:
            nbrs = neighbor_fn.setup_fn(state.positions)
        else:
            step += 1

    assert state.positions.dtype == dtype
    assert torch.all
