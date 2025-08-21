import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet.statistics

try:
    from scipy.stats import chi2_contingency

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@given(
    batch_size=st.integers(min_value=1, max_value=5),
    dtype=st.sampled_from([torch.float32, torch.float64]),
)
@settings(deadline=None)
def test_phi_coefficient(batch_size, dtype):
    """Test phi coefficient calculation."""
    # Generate test parameters
    chi_square_values = (
        torch.tensor([1.0, 4.0, 9.0], dtype=dtype).repeat(batch_size, 1).flatten()
    )
    sample_sizes = (
        torch.tensor([50, 100, 200], dtype=dtype).repeat(batch_size, 1).flatten()
    )

    # Test basic functionality
    result = beignet.statistics.phi_coefficient(chi_square_values, sample_sizes)
    assert result.shape == chi_square_values.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)

    # Test with out parameter
    out = torch.empty_like(chi_square_values)
    result_out = beignet.statistics.phi_coefficient(
        chi_square_values, sample_sizes, out=out
    )
    assert torch.allclose(result_out, out)
    assert torch.allclose(result_out, result)

    # Test that phi increases with chi-square for fixed sample size
    small_chi_sq = torch.tensor(2.0, dtype=dtype)
    large_chi_sq = torch.tensor(10.0, dtype=dtype)
    sample_size = torch.tensor(100.0, dtype=dtype)

    phi_small = beignet.statistics.phi_coefficient(small_chi_sq, sample_size)
    phi_large = beignet.statistics.phi_coefficient(large_chi_sq, sample_size)

    assert phi_large > phi_small

    # Test that phi decreases with sample size for fixed chi-square
    chi_sq = torch.tensor(6.0, dtype=dtype)
    small_n = torch.tensor(50.0, dtype=dtype)
    large_n = torch.tensor(200.0, dtype=dtype)

    phi_small_n = beignet.statistics.phi_coefficient(chi_sq, small_n)
    phi_large_n = beignet.statistics.phi_coefficient(chi_sq, large_n)

    assert phi_small_n > phi_large_n

    # Test gradient computation
    chi_sq_grad = chi_square_values.clone().requires_grad_(True)
    n_grad = sample_sizes.clone().requires_grad_(True)
    result_grad = beignet.statistics.phi_coefficient(chi_sq_grad, n_grad)

    # Compute gradients
    loss = result_grad.sum()
    loss.backward()

    assert chi_sq_grad.grad is not None
    assert n_grad.grad is not None
    assert chi_sq_grad.grad.shape == chi_square_values.shape
    assert n_grad.grad.shape == sample_sizes.shape

    # Test torch.compile compatibility
    compiled_phi_coefficient = torch.compile(
        beignet.statistics.phi_coefficient, fullgraph=True
    )
    result_compiled = compiled_phi_coefficient(chi_square_values, sample_sizes)
    assert torch.allclose(result, result_compiled, atol=1e-6)

    # Test zero chi-square (should give phi = 0)
    zero_chi_sq = torch.tensor(0.0, dtype=dtype)
    zero_phi = beignet.statistics.phi_coefficient(zero_chi_sq, sample_size)
    assert torch.abs(zero_phi) < 1e-6

    # Test phi coefficient against known values
    # Test Cohen's interpretation guidelines
    # Small effect: φ = 0.1
    chi_sq_known = torch.tensor(1.0, dtype=dtype)  # phi = sqrt(1/100) = 0.1
    n_known = torch.tensor(100.0, dtype=dtype)

    phi = beignet.statistics.phi_coefficient(chi_sq_known, n_known)
    assert torch.abs(phi - 0.1) < 1e-6

    # Medium effect: φ = 0.3
    chi_sq_medium = torch.tensor(9.0, dtype=dtype)  # phi = sqrt(9/100) = 0.3
    phi_medium = beignet.statistics.phi_coefficient(chi_sq_medium, n_known)
    assert torch.abs(phi_medium - 0.3) < 1e-6

    # Large effect: φ = 0.5
    chi_sq_large = torch.tensor(25.0, dtype=dtype)  # phi = sqrt(25/100) = 0.5
    phi_large = beignet.statistics.phi_coefficient(chi_sq_large, n_known)
    assert torch.abs(phi_large - 0.5) < 1e-6

    # Test perfect association
    chi_sq_perfect = torch.tensor(100.0, dtype=dtype)  # phi = sqrt(100/100) = 1.0
    phi_perfect = beignet.statistics.phi_coefficient(chi_sq_perfect, n_known)
    assert torch.abs(phi_perfect - 1.0) < 1e-6

    # Test phi coefficient using scipy's chi-square calculation
    if HAS_SCIPY:
        # Create test 2x2 contingency tables
        test_tables = [
            # Balanced table
            np.array([[20, 30], [25, 25]]),
            # Unbalanced table
            np.array([[10, 40], [15, 35]]),
            # Strong association
            np.array([[40, 10], [5, 45]]),
            # Weak association
            np.array([[23, 27], [26, 24]]),
        ]

        for table in test_tables:
            # Calculate chi-square statistic using scipy
            chi2_stat, p_value, dof, expected = chi2_contingency(table)

            # Get sample size
            n = table.sum()

            # Our implementation
            chi_sq_scipy = torch.tensor(chi2_stat, dtype=dtype)
            sample_size_scipy = torch.tensor(float(n), dtype=dtype)
            beignet_result = beignet.statistics.phi_coefficient(
                chi_sq_scipy, sample_size_scipy
            )

            # Manual phi calculation using standard definition: phi = sqrt(chi²/n)
            # This is the mathematically correct definition
            manual_phi = np.sqrt(chi2_stat / n)

            # Compare with manual calculation (which is the standard definition)
            tolerance = 1e-7
            diff = abs(float(beignet_result) - manual_phi)
            assert diff < tolerance, (
                f"table:\n{table}\nbeignet={float(beignet_result):.10f}, "
                + f"manual={manual_phi:.10f}, diff={diff:.10f}"
            )

    # Test phi coefficient with manual calculation
    # Example 2x2 table:
    # |  a  |  b  |
    # |  c  |  d  |
    #
    # Manual calculation: phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
    # Chi-square calculation: chi2 = n * (ad - bc)^2 / ((a+b)(c+d)(a+c)(b+d))
    # Therefore: phi = sqrt(chi2 / n)

    # Test table: [[15, 5], [10, 20]]
    a, b, c, d = 15, 5, 10, 20
    n = a + b + c + d  # 50

    # Manual phi calculation
    numerator = a * d - b * c  # 15*20 - 5*10 = 300 - 50 = 250
    denominator = np.sqrt(
        (a + b) * (c + d) * (a + c) * (b + d)
    )  # sqrt(20*30*25*25) = sqrt(375000) ≈ 612.37
    phi_manual = numerator / denominator  # 250 / 612.37 ≈ 0.408

    # Chi-square calculation
    chi2_manual = n * (numerator**2) / ((a + b) * (c + d) * (a + c) * (b + d))
    # chi2 = 50 * 250^2 / 375000 = 50 * 62500 / 375000 = 8.333...

    # Our implementation
    chi_sq_manual = torch.tensor(chi2_manual, dtype=dtype)
    sample_size_manual = torch.tensor(float(n), dtype=dtype)
    beignet_result_manual = beignet.statistics.phi_coefficient(
        chi_sq_manual, sample_size_manual
    )

    # Should match the absolute value of manual calculation
    expected_phi = abs(phi_manual)
    tolerance = 1e-6
    diff = abs(float(beignet_result_manual) - expected_phi)
    assert diff < tolerance, (
        f"Manual: {expected_phi:.8f}, Beignet: {float(beignet_result_manual):.8f}, Diff: {diff:.8f}"
    )

    # Test edge cases for phi coefficient
    # Test with very small chi-square
    tiny_chi_sq = torch.tensor(1e-10, dtype=dtype)
    n_edge = torch.tensor(100.0, dtype=dtype)
    tiny_phi = beignet.statistics.phi_coefficient(tiny_chi_sq, n_edge)
    assert tiny_phi < 1e-5

    # Test with very large sample size
    chi_sq_edge = torch.tensor(4.0, dtype=dtype)
    large_n_edge = torch.tensor(10000.0, dtype=dtype)
    small_phi = beignet.statistics.phi_coefficient(chi_sq_edge, large_n_edge)
    assert small_phi < 0.1

    # Test dtype consistency
    chi_sq_f32 = torch.tensor(4.0, dtype=torch.float32)
    n_f64 = torch.tensor(100.0, dtype=torch.float64)
    result_mixed = beignet.statistics.phi_coefficient(chi_sq_f32, n_f64)
    assert result_mixed.dtype == torch.float64
