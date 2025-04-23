# test_entropy_solver.py

import math
import time
import warnings
from typing import Any, Dict, Tuple

import torch
from beignet import newton_raphson
from torch import Tensor
from torch.distributions import Categorical
from torch.testing import assert_close  # Use PyTorch's testing utility

# --- Test Setup (Common fixture-like setup) ---

B: int = 2000
K: int = 20
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_TAU_BOUND: float = 1e-7

print("--- Entropy Example using newton_raphson ---")
print(f"Using device: {DEVICE}")
print(f"Batch size B={B}, Categories K={K}")

# ** Logit Preprocessing **
# Goal: Create reasonably scaled logits to avoid extreme tau values
#       and improve solver stability.
# 1. Start with random standard normal logits.
# 2. Normalize the L2 norm of each logit vector to sqrt(K). This makes the 'scale'
#    consistent across the batch, independent of the random draw magnitude.
# 3. Shift logits by subtracting the maximum in each row. This makes the largest
#    logit 0, which makes the temperature values less extreme for the test cases.
logits_batch: Tensor = torch.randn(B, K, device=DEVICE, dtype=torch.float32)
logits_norm: Tensor = torch.norm(logits_batch, dim=-1, keepdim=True)
logits_batch = logits_batch * math.sqrt(K) / logits_norm.clamp(1e-6)  # Normalize
logits_batch = (
    logits_batch - torch.max(logits_batch, dim=-1, keepdim=True).values
)  # Shift
print("Logits preprocessed (normalized and shifted).")

LOG_K: Tensor = torch.log(torch.tensor(K, device=DEVICE, dtype=torch.float32))
print(f"Max possible entropy (logK): {LOG_K:.4f}")


# --- Wrappers for Entropy Functions to Match Solver API ---


def entropy_func_for_solver(
    tau_batch: Tensor, logits_batch: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Calculates softmax entropy H(tau | logits) using torch.distributions.Categorical.

    Args:
        tau_batch: Current temperature estimates (N_active,). Corresponds to 'x_batch'.
        logits_batch: Logits parameters (N_active, K). Corresponds to 'params_batch'.

    Returns:
        - Entropy values H (N_active,).
        - Probabilities p (N_active, K) as auxiliary data for the derivative.
    """
    temperature: Tensor = tau_batch.unsqueeze(-1)
    scaled_logits: Tensor = logits_batch / temperature
    dist = Categorical(logits=scaled_logits)
    entropy: Tensor = dist.entropy()
    probs: Tensor = dist.probs
    return entropy, probs


def entropy_prime_func_for_solver(
    tau_batch: Tensor, logits_batch: Tensor, probs_batch: Tensor
) -> Tensor:
    """
    Calculates the derivative dH/dtau using pre-computed probabilities for
        the Newton solver.

    Args:
        tau_batch: Current temperature estimates (N_active,). Corresponds to 'x_batch'.
        logits_batch: Logits parameters (N_active, K). Corresponds to 'params_batch'.
        probs_batch: Probabilities p(tau | logits) (N_active, K).
        This is the 'aux_data'.

    Returns:
        Derivative dH/dtau values (N_active,).
    """
    temperature: Tensor = tau_batch.unsqueeze(-1)
    E_z: Tensor = torch.sum(probs_batch * logits_batch, dim=-1, keepdim=True)
    E_z_sq: Tensor = torch.sum(probs_batch * (logits_batch**2), dim=-1, keepdim=True)
    Var_z: Tensor = torch.clamp(E_z_sq - (E_z**2), min=1e-20)
    dH_dtau: Tensor = Var_z / (temperature**3 + 1e-20)
    return dH_dtau.squeeze(-1)


# --- Test Functions (pytest style) ---


def run_entropy_test(
    test_name: str,
    target_entropy_batch: Tensor,
    solver_params: Dict[str, Any],
    verify_rtol: float = 1e-3,  # Relative tolerance for assert_close
    verify_atol: float = 1e-4,  # Absolute tolerance for assert_close
):
    """Helper function to run a single entropy test case."""
    print(f"\n--- {test_name} ---")
    min_entropy = torch.min(target_entropy_batch)
    max_entropy = torch.max(target_entropy_batch)
    print(f"Target H range: [{min_entropy:.4f}, {max_entropy.max():.4f}]")
    print(f"Solver Params: {solver_params}")

    start_time = time.time()
    # Ensure min_bound is passed correctly
    solver_params_with_defaults = {
        "func": entropy_func_for_solver,
        "f_prime": entropy_prime_func_for_solver,
        "params_batch": logits_batch,
        "y_target_batch": target_entropy_batch,
        "min_bound": MIN_TAU_BOUND,
        **solver_params,  # Override defaults with test-specific params
    }
    tau_result: Tensor = newton_raphson(**solver_params_with_defaults)
    end_time = time.time()
    print(f"Calculation time: {end_time - start_time:.4f} seconds")

    # Verification
    # Recalculate final entropy using the result tau
    final_H, _ = entropy_func_for_solver(tau_result, logits_batch)

    # Use torch.testing.assert_close for robust comparison
    try:
        assert_close(final_H, target_entropy_batch, rtol=verify_rtol, atol=verify_atol)
        print(f"Verification PASSED (rtol={verify_rtol}, atol={verify_atol})")
    except AssertionError as e:
        warnings.warn(
            f"Verification FAILED for {test_name}: {e}", UserWarning, stacklevel=2
        )
        # Print max difference for debugging even if assert fails
        max_diff = torch.max(torch.abs(final_H - target_entropy_batch)).item()
        print(
            f"Max absolute error |H(tau) - H_target|: {max_diff:.6g}"
        )  # Use 'g' for auto-precision

    assert_close(final_H, target_entropy_batch, rtol=verify_rtol, atol=verify_atol)

    print(
        f"Range of found tau values: [{tau_result.min():.6g}, {tau_result.max():.6g}]"
    )
    return tau_result  # Return result if needed


def test_lower_entropies():
    """Test with target entropies in the lower range."""
    params: Dict[str, Any] = {
        "x_init": 1e-2,
        "max_iter": 200,
        "tol": 1e-4,  # Looser tol as per original test
        "max_update_step": 5e-3,
    }
    target_entropy = torch.rand(B, device=DEVICE, dtype=torch.float32) * (
        LOG_K * 0.1
    ) + (LOG_K * 0.25)
    run_entropy_test("Test Case: Lower Entropies", target_entropy, params)


def test_mid_range_entropies():
    """Test with target entropies in the mid range."""
    params: Dict[str, Any] = {
        "x_init": 0.1,
        "max_iter": 200,
        "tol": 1e-5,
        "max_update_step": 0.05,
    }
    target_entropy = torch.rand(B, device=DEVICE, dtype=torch.float32) * (
        LOG_K * 0.1
    ) + (LOG_K * 0.45)
    run_entropy_test("Test Case: Mid-range Entropies", target_entropy, params)


def test_higher_entropies():
    """Test with target entropies in the higher range."""
    params: Dict[str, Any] = {
        "x_init": 1.0,
        "max_iter": 200,
        "tol": 1e-4,  # Looser tol as per original test
        "max_update_step": 0.5,
    }
    target_entropy = torch.rand(B, device=DEVICE, dtype=torch.float32) * (
        LOG_K * 0.1
    ) + (LOG_K * 0.80)
    run_entropy_test(
        "Test Case: Higher Entropies", target_entropy, params, verify_atol=5e-4
    )  # Slightly looser verify tolerance


# --- Execute Tests (if run as script) ---
if __name__ == "__main__":
    # Note: If using pytest, you'd typically run `pytest test_entropy_solver.py`
    # This block allows running `python test_entropy_solver.py` directly.
    test_lower_entropies()
    test_mid_range_entropies()
    test_higher_entropies()
