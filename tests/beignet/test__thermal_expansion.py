import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet


@given(
    batch_size=st.integers(min_value=1, max_value=10),
    num_timesteps=st.integers(min_value=100, max_value=1000),
    dtype=st.sampled_from([torch.float32, torch.float64]),
    mean_volume=st.floats(min_value=1000.0, max_value=10000.0),
    volume_std=st.floats(min_value=1.0, max_value=100.0),
    mean_enthalpy=st.floats(min_value=-10000.0, max_value=-1000.0),
    enthalpy_std=st.floats(min_value=10.0, max_value=1000.0),
    temperature=st.floats(min_value=100.0, max_value=1000.0),
)
@settings(deadline=None)  # Disable deadline due to torch.compile
def test_thermal_expansion(
    batch_size: int,
    num_timesteps: int,
    dtype: torch.dtype,
    mean_volume: float,
    volume_std: float,
    mean_enthalpy: float,
    enthalpy_std: float,
    temperature: float,
) -> None:
    """Test thermal_expansion operator."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test single trajectory case
    volumes_single = torch.normal(
        mean=mean_volume, std=volume_std, size=(num_timesteps,), dtype=dtype
    )
    enthalpies_single = torch.normal(
        mean=mean_enthalpy, std=enthalpy_std, size=(num_timesteps,), dtype=dtype
    )
    temp_single = torch.tensor(temperature, dtype=dtype)

    result_single = beignet.thermal_expansion(
        volumes_single, temp_single, enthalpies_single
    )

    # Basic checks
    assert result_single.dtype == dtype
    assert result_single.shape == torch.Size([])
    assert torch.isfinite(result_single), "Result should be finite"

    # Test batch case
    volumes_batch = torch.normal(
        mean=mean_volume, std=volume_std, size=(batch_size, num_timesteps), dtype=dtype
    )
    enthalpies_batch = torch.normal(
        mean=mean_enthalpy,
        std=enthalpy_std,
        size=(batch_size, num_timesteps),
        dtype=dtype,
    )
    temp_batch = torch.full((batch_size,), temperature, dtype=dtype)

    result_batch = beignet.thermal_expansion(
        volumes_batch, temp_batch, enthalpies_batch
    )

    # Batch checks
    assert result_batch.dtype == dtype
    assert result_batch.shape == torch.Size([batch_size])
    assert torch.all(torch.isfinite(result_batch)), "All results should be finite"

    # Test broadcasting temperature
    result_broadcast = beignet.thermal_expansion(
        volumes_batch, temp_single, enthalpies_batch
    )
    assert result_broadcast.shape == torch.Size([batch_size])

    # Test different temperature per batch
    temp_varied = torch.linspace(200, 800, batch_size, dtype=dtype)
    result_varied = beignet.thermal_expansion(
        volumes_batch, temp_varied, enthalpies_batch
    )
    assert result_varied.shape == torch.Size([batch_size])

    # Test physical reasonableness
    # Thermal expansion coefficient should typically be positive for most materials
    # but can be negative (e.g., water between 0-4°C)
    # Typical values: 10^-6 to 10^-4 K^-1 for solids

    # Test with uncorrelated volumes and enthalpies (should give near-zero expansion)
    volumes_uncorr = torch.normal(
        mean=mean_volume, std=volume_std, size=(num_timesteps,), dtype=dtype
    )
    # Create uncorrelated enthalpies by shuffling
    enthalpies_uncorr = enthalpies_single[torch.randperm(num_timesteps)]
    result_uncorr = beignet.thermal_expansion(
        volumes_uncorr, temp_single, enthalpies_uncorr
    )
    assert torch.isfinite(result_uncorr)

    # Test numerical stability with constant values (no fluctuations)
    volumes_constant = torch.full((num_timesteps,), mean_volume, dtype=dtype)
    enthalpies_constant = torch.full((num_timesteps,), mean_enthalpy, dtype=dtype)
    result_constant = beignet.thermal_expansion(
        volumes_constant, temp_single, enthalpies_constant
    )
    # With no fluctuations, thermal expansion should be zero
    if dtype == torch.float64:
        assert torch.allclose(
            result_constant, torch.tensor(0.0, dtype=dtype), atol=1e-10
        ), "Zero fluctuations should give zero thermal expansion"
    else:
        # float32 has lower precision
        assert torch.allclose(
            result_constant, torch.tensor(0.0, dtype=dtype), atol=1e-5
        ), "Zero fluctuations should give near-zero thermal expansion"

    # Test edge case with very small number of timesteps
    if num_timesteps >= 2:
        volumes_small = volumes_single[:2]
        enthalpies_small = enthalpies_single[:2]
        result_small = beignet.thermal_expansion(
            volumes_small, temp_single, enthalpies_small
        )
        assert torch.isfinite(result_small)

    # Test gradient computation
    if dtype == torch.float64:
        volumes_grad = volumes_single.clone().requires_grad_(True)
        enthalpies_grad = enthalpies_single.clone().requires_grad_(True)
        temp_grad = temp_single.clone().requires_grad_(True)

        result_grad = beignet.thermal_expansion(
            volumes_grad, temp_grad, enthalpies_grad
        )
        result_grad.backward()

        assert volumes_grad.grad is not None
        assert enthalpies_grad.grad is not None
        assert temp_grad.grad is not None
        assert torch.all(torch.isfinite(volumes_grad.grad))
        assert torch.all(torch.isfinite(enthalpies_grad.grad))
        assert torch.isfinite(temp_grad.grad)

    # Test torch.compile compatibility
    compiled_fn = torch.compile(beignet.thermal_expansion, fullgraph=True)
    result_compiled = compiled_fn(volumes_single, temp_single, enthalpies_single)
    # Compiled version may have different numerical precision
    assert torch.allclose(result_single, result_compiled, rtol=1e-4, atol=1e-6)

    # Test vmap compatibility
    from torch.func import vmap

    # Single trajectory, multiple temperatures
    temps_vmap = torch.linspace(200, 800, 10, dtype=dtype)
    vmap_fn = vmap(
        lambda t: beignet.thermal_expansion(volumes_single, t, enthalpies_single)
    )
    result_vmap = vmap_fn(temps_vmap)
    assert result_vmap.shape == torch.Size([10])
    assert torch.all(torch.isfinite(result_vmap))

    # Test relationship with temperature
    # α ∝ 1/T² (from the formula)
    if batch_size >= 2 and temperature > 200:
        temp_low = torch.tensor(temperature, dtype=dtype)
        temp_high = torch.tensor(temperature * 2, dtype=dtype)

        # Use same data for both calculations
        result_low = beignet.thermal_expansion(
            volumes_single, temp_low, enthalpies_single
        )
        result_high = beignet.thermal_expansion(
            volumes_single, temp_high, enthalpies_single
        )

        # For same correlation, α should scale as 1/T²
        # α_low / α_high ≈ (T_high / T_low)²
        if torch.abs(result_low) > 1e-10 and torch.abs(result_high) > 1e-10:
            ratio = result_low / result_high
            expected_ratio = (temp_high / temp_low) ** 2
            # Allow for some tolerance due to numerical precision
            assert torch.allclose(ratio, expected_ratio, rtol=0.1)

    # Test units consistency
    # Result should be in K^-1 (per Kelvin)
    # Typical values for solids: 10^-6 to 10^-4 K^-1
    # The actual magnitude depends on the units of input data
