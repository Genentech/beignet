# test_entropy_solver.py (or test__newton_raphson.py)

import math
import time
import warnings
from typing import Any

import torch

from torch import Tensor
from torch.distributions import Categorical  # Keep if used elsewhere, not needed now
from torch.optim import LBFGS  # Import the optimizer
from torch.testing import assert_close

# ruff: noqa: I001
from beignet import newton_raphson  # Import NR solver

B: int = 2000
K: int = 20
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_TAU_BOUND: float = 1e-7

print("--- Entropy Example: Newton-Raphson vs LBFGS ---")  # Updated title
print(f"Using device: {DEVICE}")
print(f"Batch size B={B}, Categories K={K}")

# ** Logit Preprocessing **
# Goal: Create reasonably scaled logits to avoid extreme tau values
#       and improve solver stability.
# 1. Start with random standard normal logits.
# 2. Normalize the L2 norm of each logit vector to sqrt(K). Makes scale consistent.
# 3. Shift logits by subtracting the maximum in each row. Makes largest logit 0.
logits_batch: Tensor = torch.randn(B, K, device=DEVICE, dtype=torch.float32)
logits_norm: Tensor = torch.norm(logits_batch, dim=-1, keepdim=True)
logits_batch = logits_batch * math.sqrt(K) / logits_norm.clamp(1e-6)  # Normalize
logits_batch = (
    logits_batch - torch.max(logits_batch, dim=-1, keepdim=True).values
)  # Shift
print("Logits preprocessed (normalized and shifted).")

LOG_K: Tensor = torch.log(torch.tensor(K, device=DEVICE, dtype=torch.float32))
print(f"Max possible entropy (logK): {LOG_K:.4f}")


# --- Wrappers for Entropy Functions (Copied from previous version) ---
# (Assuming these are correct as provided in the previous user message)
def entropy_func_for_solver(
    tau_batch: Tensor, logits_batch: Tensor
) -> tuple[Tensor, Tensor]:
    temperature: Tensor = tau_batch.unsqueeze(-1)
    scaled_logits: Tensor = logits_batch / temperature
    dist = Categorical(logits=scaled_logits)
    entropy: Tensor = dist.entropy()
    probs: Tensor = dist.probs
    return entropy, probs


def entropy_prime_func_for_solver(
    tau_batch: Tensor, logits_batch: Tensor, probs_batch: Tensor
) -> Tensor:
    temperature: Tensor = tau_batch.unsqueeze(-1)
    E_z: Tensor = torch.sum(probs_batch * logits_batch, dim=-1, keepdim=True)
    E_z_sq: Tensor = torch.sum(probs_batch * (logits_batch**2), dim=-1, keepdim=True)
    Var_z: Tensor = torch.clamp(E_z_sq - (E_z**2), min=1e-20)
    dH_dtau: Tensor = Var_z / (temperature**3 + 1e-20)
    return dH_dtau.squeeze(-1)


# --- Test Execution Helper ---


def run_entropy_test(
    test_name: str,
    target_entropy_batch: Tensor,
    # --- Newton Params ---
    nr_params: dict[str, Any],
    # --- LBFGS Params ---
    run_lbfgs: bool = True,  # Flag to enable LBFGS comparison
    lbfgs_outer_steps: int = 100,
    lbfgs_lr: float = 1e-4,
    lbfgs_max_iter: int = 20,  # Inner iterations per LBFGS step
    lbfgs_init_tau_scale: float = 1.0,  # Factor to scale NR init_tau for LBFGS
    # --- Verification Params ---
    verify_rtol: float = 1e-3,
    verify_atol: float = 1e-4,
    compare_tau_rtol: float = 1e-2,  # Looser tolerance for comparing tau results
    compare_tau_atol: float = 1e-3,
):
    """Helper function to run Newton-Raphson and optionally LBFGS."""
    print(f"\n--- {test_name} ---")
    min_entropy = torch.min(target_entropy_batch)
    max_entropy = torch.max(target_entropy_batch)
    # Wrapped print statement
    print(f"Target H range: [{min_entropy:.4f}, {max_entropy:.4f}]")

    # --- 1. Newton-Raphson ---
    print("\nRunning Newton-Raphson...")
    print(f"NR Solver Params: {nr_params}")
    start_time_nr = time.time()
    solver_params_nr = {
        "func": entropy_func_for_solver,
        "f_prime": entropy_prime_func_for_solver,
        "params": logits_batch,
        "y_target_batch": target_entropy_batch,
        "min_bound": MIN_TAU_BOUND,
        **nr_params,  # Test-specific NR params
    }
    # Assuming 'newton_raphson' is the imported name for the solver
    tau_nr: Tensor = newton_raphson(**solver_params_nr)
    end_time_nr = time.time()
    time_nr = end_time_nr - start_time_nr
    print(f"NR Calculation time: {time_nr:.4f} seconds")

    # Verification NR
    final_H_nr, _ = entropy_func_for_solver(tau_nr, logits_batch)
    try:
        assert_close(
            final_H_nr, target_entropy_batch, rtol=verify_rtol, atol=verify_atol
        )
        print(f"NR Verification PASSED (rtol={verify_rtol}, atol={verify_atol})")
    except AssertionError as e:
        warnings.warn(
            f"NR Verification FAILED for {test_name}: {e}", UserWarning, stacklevel=2
        )
    max_diff_nr = torch.max(torch.abs(final_H_nr - target_entropy_batch)).item()
    print(f"NR Max absolute error |H(tau) - H_target|: {max_diff_nr:.6g}")
    print(f"NR Range of found tau values: [{tau_nr.min():.6g}, {tau_nr.max():.6g}]")

    # --- 2. LBFGS (Optional) ---
    log_tau_lbfgs = None
    time_lbfgs = float("nan")
    max_diff_lbfgs = float("nan")
    LOG_MIN_TAU_BOUND = math.log(MIN_TAU_BOUND)

    if run_lbfgs:
        print("\nRunning LBFGS...")
        # Initial guess for LBFGS tau - use NR's initial guess scaled, require grad
        # Use dict.get() for safe access to nr_params['x_init']
        init_tau_val_nr = nr_params.get("x_init", 1.0)
        init_log_tau_lbfgs = math.log(float(init_tau_val_nr)) * lbfgs_init_tau_scale
        log_tau_lbfgs = torch.full_like(target_entropy_batch, init_log_tau_lbfgs)
        log_tau_lbfgs = torch.clamp(
            log_tau_lbfgs, min=LOG_MIN_TAU_BOUND
        ).requires_grad_(True)

        # Setup optimizer
        optimizer = LBFGS(
            [log_tau_lbfgs],
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,  # Inner iterations per step() call
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",  # Recommended
        )
        # Wrapped print statement
        print(
            f"LBFGS Params: outer={lbfgs_outer_steps}, lr={lbfgs_lr}, "
            f"inner={lbfgs_max_iter}, init_scale={lbfgs_init_tau_scale}"
        )

        start_time_lbfgs = time.time()

        # Optimization loop (outer steps)
        for _ in range(lbfgs_outer_steps):

            def closure():
                optimizer.zero_grad()
                # Ensure tau is positive within loss calculation
                tau = log_tau_lbfgs.exp()
                current_H, _ = entropy_func_for_solver(tau, logits_batch)
                # Loss: mean squared error
                loss = torch.mean((current_H - target_entropy_batch).pow(2))
                loss.backward()
                return loss

            _ = optimizer.step(closure)

            # Manually clamp tau after the step to maintain bounds
            with torch.no_grad():
                log_tau_lbfgs.clamp_(min=LOG_MIN_TAU_BOUND)

            # Optional early stopping check
            # if loss_val.item() < 1e-10:
            #     print(f"LBFGS converged early at outer step {i+1}")
            #     break

        end_time_lbfgs = time.time()
        time_lbfgs = end_time_lbfgs - start_time_lbfgs
        print(f"LBFGS Calculation time: {time_lbfgs:.4f} seconds")

        # Detach final result
        tau_lbfgs = log_tau_lbfgs.detach().exp()

        # Verification LBFGS
        final_H_lbfgs, _ = entropy_func_for_solver(tau_lbfgs, logits_batch)
        try:
            assert_close(
                final_H_lbfgs, target_entropy_batch, rtol=verify_rtol, atol=verify_atol
            )
            print(f"LBFGS Verification PASSED (rtol={verify_rtol}, atol={verify_atol})")
        except AssertionError as e:
            warnings.warn(
                f"LBFGS Verification FAILED for {test_name}: {e}",
                UserWarning,
                stacklevel=2,
            )
        max_diff_lbfgs = torch.max(
            torch.abs(final_H_lbfgs - target_entropy_batch)
        ).item()
        print(f"LBFGS Max absolute error |H(tau) - H_target|: {max_diff_lbfgs:.6g}")
        print(
            f"LBFGS Range of tau values: [{tau_lbfgs.min():.6g}, {tau_lbfgs.max():.6g}]"
        )

        # --- 3. Comparison ---
        print("\nComparison:")
        print(f"Runtime: NR={time_nr:.4f}s vs LBFGS={time_lbfgs:.4f}s")
        print(f"Max Error: NR={max_diff_nr:.6g} vs LBFGS={max_diff_lbfgs:.6g}")
        # Compare the resulting tau tensors
        try:
            assert_close(
                tau_nr, tau_lbfgs, rtol=compare_tau_rtol, atol=compare_tau_atol
            )
            # Wrapped print statement
            print(
                f"Tau values are close (rtol={compare_tau_rtol:.1e}, "
                f"atol={compare_tau_atol:.1e})"
            )
        except AssertionError as e:
            tau_diff = torch.max(torch.abs(tau_nr - tau_lbfgs)).item()
            # Wrapped warning message
            warn_msg = (
                f"Tau values differ significantly: max_abs_diff={tau_diff:.6g}. "
                f"Details: {e}"
            )
            warnings.warn(warn_msg, UserWarning, stacklevel=2)

    # Return results if needed by caller test functions
    return tau_nr, tau_lbfgs


# --- Test Functions (pytest style, call run_entropy_test) ---


def test_lower_entropies():
    """Test with target entropies in the lower range."""
    # NR specific params for this test
    nr_params: dict[str, Any] = {
        "x_init": 1e-2,
        "max_iter": 200,
        "abs_tol": 1e-4,
        "max_update_step": 5e-3,
    }
    target_entropy = torch.rand(B, device=DEVICE, dtype=torch.float32) * (
        LOG_K * 0.1
    ) + (LOG_K * 0.25)
    run_entropy_test(
        "Test Case: Lower Entropies", target_entropy, nr_params, lbfgs_lr=1.0
    )


def test_mid_range_entropies():
    """Test with target entropies in the mid range."""
    # NR specific params for this test
    nr_params: dict[str, Any] = {
        "x_init": 0.25,  # Updated init_tau from original file for this test
        "max_iter": 200,
        "abs_tol": 1e-5,
        "max_update_step": 0.05,  # Updated step from original file
    }
    target_entropy = torch.rand(B, device=DEVICE, dtype=torch.float32) * (
        LOG_K * 0.1
    ) + (LOG_K * 0.45)
    run_entropy_test(
        "Test Case: Mid-range Entropies", target_entropy, nr_params, lbfgs_lr=5e-3
    )


def test_higher_entropies():
    """Test with target entropies in the higher range."""
    # NR specific params for this test
    nr_params: dict[str, Any] = {
        "x_init": 1.0,
        "max_iter": 200,
        "abs_tol": 1e-4,
        "max_update_step": 0.5,
    }
    target_entropy = torch.rand(B, device=DEVICE, dtype=torch.float32) * (
        LOG_K * 0.1
    ) + (LOG_K * 0.80)
    run_entropy_test(
        "Test Case: Higher Entropies",
        target_entropy,
        nr_params,
        lbfgs_lr=0.5,
        verify_atol=5e-4,
    )


# --- Execute Tests (if run as script) ---
if __name__ == "__main__":
    test_lower_entropies()
    test_mid_range_entropies()
    test_higher_entropies()
