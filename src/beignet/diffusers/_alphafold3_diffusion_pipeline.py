# af3_pipeline.py
from __future__ import annotations

from typing import Optional

import torch
from diffusers import DiffusionPipeline
from torch import Tensor, nn

from .schedulers import AlphaFold3Scheduler


class AlphaFold3DiffusionPipeline(DiffusionPipeline):
    r"""
    A small Diffusers pipeline for AF3-style coordinate diffusion.

    Modules:
        trunk         : your AF3 trunk (must expose `encode_conditioners(f_star)`
                         -> (s_inputs, s_trunk, z_trunk))
        diffusion     : your DiffusionModule (x_noisy, t, f_star_pos, s_inputs, s_trunk, z_trunk, z_atom) -> x_denoised
        scheduler     : AF3Scheduler
        centre_aug    : your CentreRandomAugmentation (x) -> x

    Call:
        __call__(f_star, schedule=None, z_atom_dim=16) -> x_final (B,N,3)
    """

    def __init__(
        self,
        trunk: nn.Module,
        diffusion: nn.Module,
        scheduler: AlphaFold3Scheduler,
        centre_aug: nn.Module,
    ):
        super().__init__()
        self.register_modules(
            trunk=trunk,
            diffusion=diffusion,
            scheduler=scheduler,
            centre_aug=centre_aug,
        )

    @torch.no_grad()
    def __call__(
        self,
        f_star: dict[str, Tensor],
        *,
        schedule: Optional[Tensor] = None,
        z_atom_dim: int = 16,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Args:
            f_star: dict with at least 'ref_pos' (B,N,3). Other features are trunk-specific.
            schedule: optional (T+1,) tensor to override scheduler.c
            z_atom_dim: channel dimension for per-atom pair features (zeros by default)
            generator: torch.Generator for reproducible sampling
            dtype: compute dtype override

        Returns:
            x: (B,N,3) final coordinates
        """
        device = self.device
        if dtype is None:
            dtype = next(self.trunk.parameters()).dtype

        # Optionally override scheduler schedule for this call
        if schedule is not None:
            assert schedule.ndim == 1 and schedule.numel() >= 2
            self.scheduler.c = torch.as_tensor(
                schedule, device=device, dtype=torch.float32
            )

        # Encode conditioners once (no diffusion inside)
        s_inputs, s_trunk, z_trunk = self.trunk.encode_conditioners(
            f_star
        )  # user-provided hook
        B, N, _ = f_star["ref_pos"].shape
        ref_pos = f_star["ref_pos"].to(device=device, dtype=dtype)

        # Init x_0 ~ c_0 * N(0, I)
        c0 = self.scheduler.c[0].to(device)
        x = c0 * torch.randn((B, N, 3), device=device, dtype=dtype, generator=generator)

        # Pair features for atoms (zeros by default; plug yours if available)
        z_atom = torch.zeros(B, N, N, z_atom_dim, device=device, dtype=dtype)

        # Main loop over τ = 1..T
        T = self.scheduler.num_inference_steps
        for t_idx in range(1, T + 1):
            # 1) centre-random augmentation
            x = self.centre_aug(x)

            # 2) construct x_noisy and t_hat from x (teacher-free during sampling)
            #    (This matches your SampleDiffusion forward; we don't use clean coords here.)
            #    We replicate the same ζ build inside the diffusion call by reusing the scheduler logic.
            c = self.scheduler.c
            c_tau = c[t_idx].to(device)
            c_prev = c[t_idx - 1].to(device)
            gamma = (
                self.scheduler.config.gamma0
                if float(c_tau) > self.scheduler.config.gamma_min
                else 0.0
            )
            t_hat = c_prev * (gamma + 1.0)

            variance = (t_hat**2 - c_prev**2).clamp_min(0.0)
            zeta = (
                self.scheduler.config.noise_scale
                * variance.sqrt()
                * torch.randn_like(x, generator=generator)
            )
            x_noisy = x + zeta

            # 3) run the diffusion model to get x̂ at t̂
            t_tensor = t_hat.expand(B, 1, 1).to(device=device, dtype=dtype)
            x_denoised = self.diffusion(
                x_noisy=x_noisy,
                t=t_tensor,
                f_star=ref_pos,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                z_atom=z_atom,
            )

            # 4) AF3 update step
            out = self.scheduler.step(
                x_denoised=x_denoised,
                x_noisy=x_noisy,
                x_prev_ref=x,
                t_index=t_idx,
            )
            x = out.prev_sample

        return x
