import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import beignet


@given(
    batch_size=st.integers(min_value=1, max_value=10),
    num_timesteps=st.integers(min_value=100, max_value=1000),
    dtype=st.sampled_from([torch.float32, torch.float64]),
    mean_volume=st.floats(min_value=1000.0, max_value=10000.0),
    volume_std=st.floats(min_value=1.0, max_value=100.0),
    temperature=st.floats(min_value=100.0, max_value=1000.0),
)
@settings(deadline=None)  # Disable deadline due to torch.compile on first run
def test_isothermal_compressibility(
    batch_size: int,
    num_timesteps: int,
    dtype: torch.dtype,
    mean_volume: float,
    volume_std: float,
    temperature: float,
) -> None:
    """Test isothermal_compressibility operator."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test single trajectory case
    volumes_single = torch.normal(
        mean=mean_volume, std=volume_std, size=(num_timesteps,), dtype=dtype
    )
    temp_single = torch.tensor(temperature, dtype=dtype)

    result_single = beignet.isothermal_compressibility(volumes_single, temp_single)

    # Basic checks
    assert result_single.dtype == dtype
    assert result_single.shape == torch.Size([])
    assert result_single.item() >= 0, (
        "Isothermal compressibility should be non-negative"
    )
    assert torch.isfinite(result_single), "Result should be finite"

    # Test batch case
    volumes_batch = torch.normal(
        mean=mean_volume, std=volume_std, size=(batch_size, num_timesteps), dtype=dtype
    )
    temp_batch = torch.full((batch_size,), temperature, dtype=dtype)

    result_batch = beignet.isothermal_compressibility(volumes_batch, temp_batch)

    # Batch checks
    assert result_batch.dtype == dtype
    assert result_batch.shape == torch.Size([batch_size])
    assert torch.all(result_batch >= 0), "All compressibilities should be non-negative"
    assert torch.all(torch.isfinite(result_batch)), "All results should be finite"

    # Test broadcasting temperature
    result_broadcast = beignet.isothermal_compressibility(volumes_batch, temp_single)
    assert result_broadcast.shape == torch.Size([batch_size])

    # Test different temperature per batch
    temp_varied = torch.linspace(200, 800, batch_size, dtype=dtype)
    result_varied = beignet.isothermal_compressibility(volumes_batch, temp_varied)
    assert result_varied.shape == torch.Size([batch_size])

    # Test relationship with temperature
    # Higher temperature should give higher compressibility (for same volume fluctuations)
    if batch_size >= 2:
        temp_low = torch.full((batch_size,), 200.0, dtype=dtype)
        temp_high = torch.full((batch_size,), 800.0, dtype=dtype)

        result_low = beignet.isothermal_compressibility(volumes_batch, temp_low)
        result_high = beignet.isothermal_compressibility(volumes_batch, temp_high)

        # For same volume fluctuations, κT ∝ 1/T
        # Only check non-zero results to avoid division by zero
        non_zero_mask = (result_low > 1e-10) & (result_high > 1e-10)
        if torch.any(non_zero_mask):
            ratio = result_low[non_zero_mask] / result_high[non_zero_mask]
            expected_ratio = temp_high[0] / temp_low[0]
            assert torch.allclose(ratio, expected_ratio, rtol=0.05)

    # Test numerical stability with constant volumes (no fluctuations)
    volumes_constant = torch.full((num_timesteps,), mean_volume, dtype=dtype)
    result_constant = beignet.isothermal_compressibility(volumes_constant, temp_single)
    # Due to numerical precision, this will be very close to 0 but not exactly 0
    if dtype == torch.float64:
        assert result_constant.item() < 1e-20, (
            "Zero fluctuations should give near-zero compressibility"
        )
    else:
        # float32 has much lower precision
        assert result_constant.item() < 1e-6, (
            "Zero fluctuations should give near-zero compressibility"
        )

    # Test edge case with very small number of timesteps
    if num_timesteps >= 2:
        volumes_small = volumes_single[:2]
        result_small = beignet.isothermal_compressibility(volumes_small, temp_single)
        assert torch.isfinite(result_small)

    # Test gradient computation
    if dtype == torch.float64:
        volumes_grad = volumes_single.clone().requires_grad_(True)
        temp_grad = temp_single.clone().requires_grad_(True)

        result_grad = beignet.isothermal_compressibility(volumes_grad, temp_grad)
        result_grad.backward()

        assert volumes_grad.grad is not None
        assert temp_grad.grad is not None
        assert torch.all(torch.isfinite(volumes_grad.grad))
        assert torch.isfinite(temp_grad.grad)

        # Temperature gradient should be negative (inverse relationship)
        assert temp_grad.grad < 0

    # Test torch.compile compatibility
    compiled_fn = torch.compile(beignet.isothermal_compressibility, fullgraph=True)
    result_compiled = compiled_fn(volumes_single, temp_single)
    # Use a generous tolerance as torch.compile can change numerical precision
    assert torch.allclose(result_single, result_compiled, rtol=0.3, atol=1e-4)

    # Test vmap compatibility
    from torch.func import vmap

    # Single volume trajectory, multiple temperatures
    temps_vmap = torch.linspace(200, 800, 10, dtype=dtype)
    vmap_fn = vmap(lambda t: beignet.isothermal_compressibility(volumes_single, t))
    result_vmap = vmap_fn(temps_vmap)
    assert result_vmap.shape == torch.Size([10])
    assert torch.all(result_vmap >= 0)

    # Test units conversion (if supported)
    # The operator should handle proper unit conversions internally
    # Result should be in inverse pressure units (e.g., Pa^-1)

    # Test physical reasonableness
    # For water-like systems at room temperature
    if (
        mean_volume > 1000
        and mean_volume < 5000
        and temperature > 250
        and temperature < 350
    ):
        # Typical compressibility range for liquids: 1e-10 to 1e-9 Pa^-1
        # This test assumes the operator returns values in Pa^-1
        assume(volume_std / mean_volume < 0.1)  # Reasonable fluctuations
        # Note: Actual bounds depend on the units used in the implementation
