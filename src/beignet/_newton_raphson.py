# _newton_raphson.py
import warnings
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import Tensor

# Define function signature types
FuncType = Callable[[Tensor, Tensor], Tuple[Tensor, Any]]
FPrimeType = Callable[[Tensor, Tensor, Any], Tensor]


def newton_raphson(
    func: FuncType,
    f_prime: FPrimeType,
    params_batch: Tensor,
    y_target_batch: Tensor,
    x_init: Union[float, Tensor],
    max_iter: int = 50,
    tol: float = 1e-5,
    min_bound: Optional[float] = None,
    max_bound: Optional[float] = None,
    max_update_step: float = 1.0,
) -> Tensor:
    """
    Performs batched Newton-Raphson root finding: finds x such that
    func(x, params) = y_target. # <-- Wrapped line 26

    Uses update step clamping for stability.

    Args:
        func: A function `f(x_batch, params_batch)` -> (y_batch, aux_data).
        f_prime: A function `f'(x_batch, params_batch, aux_data)` -> y_prime_batch.
        params_batch: Batch of parameters specific to each function instance. (B, ...).
        y_target_batch: Target values for each instance (B,).
        x_init: Initial guess for x. Scalar or Tensor (B,).
        max_iter: Maximum number of iterations.
        tol: Absolute tolerance for convergence check (|y_val - y_target| < tol).
        min_bound: Lower bound to clamp x during iteration.
        max_bound: Upper bound to clamp x during iteration.
        max_update_step: Maximum absolute value allowed for the x update step.
                         Must be positive. # <-- Wrapped line 40

    Returns:
        The found roots x_batch (B,). Non-converged elements have last computed value.
    """
    B: int = params_batch.shape[0]
    device: torch.device = params_batch.device
    dtype: torch.dtype = params_batch.dtype  # Use dtype from params for consistency

    if not torch.is_tensor(y_target_batch):
        y_target_batch = torch.tensor(y_target_batch, device=device, dtype=dtype)
    if y_target_batch.ndim == 0:  # If single target, expand to batch
        y_target_batch = y_target_batch.expand(B)
    if y_target_batch.shape[0] != B:
        raise ValueError("y_target_batch must have shape (B,) or be a scalar.")

    if max_update_step <= 0:
        raise ValueError("max_update_step must be positive.")

    # Initialize x
    x: Tensor
    if isinstance(x_init, torch.Tensor):
        x = x_init.clone().to(device, dtype)
        if x.shape[0] != B:
            raise ValueError(
                f"x_init tensor shape {x.shape} incompatible with batch size {B}"
            )
    else:
        # Ensure x is created with the correct dtype
        x = torch.full_like(y_target_batch, float(x_init), dtype=dtype)

    # Ensure initial x respects bounds
    if min_bound is not None:
        x = torch.clamp(x, min=min_bound)
    if max_bound is not None:
        x = torch.clamp(x, max=max_bound)

    # Track active elements (those not yet converged)
    active: Tensor = torch.ones_like(y_target_batch, dtype=torch.bool)

    for i in range(max_iter):
        if not torch.any(active):
            break

        # --- Operate only on active elements ---
        active_indices: Tensor = torch.where(active)[0]
        # Ensure active_indices isn't empty before proceeding
        if active_indices.numel() == 0:
            break
        x_active: Tensor = x[active]
        params_active: Tensor = params_batch[active]
        y_target_active: Tensor = y_target_batch[active]

        # --- Calculate func and f_prime ---
        try:
            y_val: Tensor
            aux_data: Any
            y_prime_val: Tensor
            y_val, aux_data = func(x_active, params_active)
            y_prime_val = f_prime(x_active, params_active, aux_data)
        except RuntimeError as e:
            warnings.warn(
                f"RuntimeError at (iter {i}): {e}. Check inputs/x values.",
                UserWarning,
                stacklevel=2,
            )
            y_prime_val = torch.ones_like(x_active)

        # --- Calculate Error ---
        g_val: Tensor = y_val - y_target_active

        # --- Convergence Check (Absolute Tolerance) ---
        converged_now: Tensor = torch.abs(g_val) < tol
        active[active_indices[converged_now]] = False  # Deactivate converged elements

        if not torch.any(active):  # Break if all converged this iteration
            break

        # --- Update Step (for non-converged elements) ---
        update_mask: Tensor = ~converged_now
        if not torch.any(update_mask):
            continue  # Skip update if none left

        active_indices_update: Tensor = active_indices[update_mask]
        if active_indices_update.numel() == 0:
            continue

        g_val_update: Tensor = g_val[update_mask]
        y_prime_val_update: Tensor = y_prime_val[update_mask]

        # Newton Step: update = error / derivative
        raw_update: Tensor = g_val_update / (y_prime_val_update + 1e-12)

        # Clamp the magnitude of the update step
        clamped_update: Tensor = torch.clamp(
            raw_update, min=-max_update_step, max=max_update_step
        )

        # Ensure indices align before subtraction
        x_active_update: Tensor = x_active[update_mask]
        x_new: Tensor = x_active_update - clamped_update

        # Update main x tensor, clamping to bounds
        if min_bound is not None:
            x_new = torch.clamp(x_new, min=min_bound)
        if max_bound is not None:
            x_new = torch.clamp(x_new, max=max_bound)
        x[active_indices_update] = x_new

    # --- Final Warning for Non-Convergence ---
    if i == max_iter - 1 and torch.any(active):
        num_not_converged: int = torch.sum(active).item()

        # Define the message template using placeholders
        warning_template = (
            "{num}/{total} elements did not converge within {iters} iterations "
            "(using max_update_step={max_update}). "
            "The returned x for these elements is the estimate after {iters} steps. "
            "Consider increasing max_iter, adjusting tol/max_update_step, "
            "checking target values, or examining the corresponding parameters."
        )

        # Format the message with the variables
        warning_message = warning_template.format(
            num=num_not_converged, total=B, iters=max_iter, max_update=max_update_step
        )

        # Call warnings.warn with the formatted message
        warnings.warn(
            warning_message,
            UserWarning,
            stacklevel=2,
        )

    return x
