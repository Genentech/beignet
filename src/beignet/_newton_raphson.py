import warnings
from typing import Any, Callable, Union

import torch
from torch import Tensor

# Define function signature types
_Func = Callable[[Tensor, Tensor], tuple[Tensor, Any]]
_FPrime = Callable[[Tensor, Tensor, Any], Tensor]


def newton_raphson(
    func: _Func,
    f_prime: _FPrime,
    params: Tensor,
    y_target_batch: Tensor,
    x_init: Union[float, Tensor],
    max_iter: int = 50,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-5,
    min_bound: float | Tensor | None = None,
    max_bound: float | Tensor | None = None,
    max_update_step: float = 1.0,
) -> Tensor:
    """
    Performs batched Newton-Raphson root finding: finds x such that
    func(x, params) = y_target. # <-- Wrapped line 26

    Uses update step clamping for stability.

    Args:
        func: A function `f(x_batch, params)` -> (y_batch, aux_data).
        f_prime: A function `f'(x_batch, params, aux_data)` -> y_prime_batch.
        params: Batch of parameters specific to each function instance. (B, ...).
        y_target_batch: Target values for each instance (B,).
        x_init: Initial guess for x. Scalar or Tensor (B,).
        max_iter: Maximum number of iterations.
        abs_tol: Abs tolerance for convergence check
            (|y_val - y_target| < abs_tol).
        rel_tol: Rel tolerance for convergence check
            (|y_val - y_target| < rel_tol * |y_target|).
        min_bound: Lower bound to clamp x during iteration.
        max_bound: Upper bound to clamp x during iteration.
        max_update_step: Maximum absolute value allowed for the x update step.
                         Must be positive.

    Returns:
        The found roots x_batch (B,). Non-converged elements have last computed value.
    """
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative.")
    if max_iter == 0:
        return x_init

    batch_size = params.shape[0]
    device = params.device
    dtype = params.dtype  # Use dtype from params for consistency

    if not torch.is_tensor(y_target_batch):
        y_target_batch = torch.tensor(y_target_batch, device=device, dtype=dtype)

    if y_target_batch.ndim == 0:  # If single target, expand to batch
        y_target_batch = y_target_batch.expand(batch_size)

    if y_target_batch.shape[0] != batch_size:
        raise ValueError("y_target_batch must have shape (B,) or be a scalar.")

    if max_update_step <= 0:
        raise ValueError("max_update_step must be positive.")

    if isinstance(x_init, torch.Tensor):
        x = x_init.clone().to(device, dtype)
        if x.shape[0] != batch_size:
            raise ValueError(
                f"x_init shape {x.shape} incompatible with params shape {params.shape}."
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
    active = torch.ones_like(y_target_batch, dtype=torch.bool)

    for _ in range(max_iter):
        if not torch.any(active):
            break

        # --- Operate only on active elements ---
        active_indices = torch.where(active)[0]
        # Ensure active_indices isn't empty before proceeding
        if active_indices.numel() == 0:
            break
        x_active = x[active]
        params_active = params[active]
        y_target_active = y_target_batch[active]

        # --- Calculate func and f_prime ---
        y_val, aux_data = func(x_active, params_active)
        y_prime_val = f_prime(x_active, params_active, aux_data)

        # --- Calculate Error ---
        g_val = y_val - y_target_active

        # --- Convergence Check (Absolute Tolerance) ---
        converged_now_abs = torch.abs(g_val) < abs_tol
        converged_now_rel = torch.abs(g_val) < rel_tol * torch.abs(y_target_active)
        converged_now = converged_now_abs & converged_now_rel
        active[active_indices[converged_now]] = False  # Deactivate converged elements

        if not torch.any(active):  # Break if all converged this iteration
            break

        # --- Update Step (for non-converged elements) ---
        update_mask = ~converged_now
        if not torch.any(update_mask):
            continue  # Skip update if none left

        active_indices_update = active_indices[update_mask]
        if active_indices_update.numel() == 0:
            continue

        g_val_update = g_val[update_mask]
        y_prime_val_update = y_prime_val[update_mask]

        # Newton Step: update = error / derivative
        raw_update = g_val_update / (y_prime_val_update + 1e-7)

        # Clamp the magnitude of the update step
        clamped_update = torch.clamp(
            raw_update, min=-max_update_step, max=max_update_step
        )

        # Ensure indices align before subtraction
        x_active_update = x_active[update_mask]
        x_new = x_active_update - clamped_update

        # Update main x tensor, clamping to bounds
        if min_bound is not None:
            x_new = torch.clamp(x_new, min=min_bound)
        if max_bound is not None:
            x_new = torch.clamp(x_new, max=max_bound)
        x[active_indices_update] = x_new

    # --- Final Warning for Non-Convergence ---
    if torch.any(active):
        num_not_converged: int = torch.sum(active).item()

        warning_template = (
            "{num}/{total} elements did not converge within {iters} iterations "
            "(using max_update_step={max_update}). "
            "The returned x for these elements is the estimate after {iters} steps. "
            "Consider increasing max_iter, adjusting abs_tol/max_update_step, "
            "checking target values, or examining the corresponding parameters."
        )

        warning_message = warning_template.format(
            num=num_not_converged,
            total=batch_size,
            iters=max_iter,
            max_update=max_update_step,
        )

        warnings.warn(
            warning_message,
            UserWarning,
            stacklevel=2,
        )

    return x
