from dataclasses import dataclass

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from torch import Tensor


@dataclass
class AF3SchedulerOutput(SchedulerOutput):
    # (inherits prev_sample: Tensor)
    pass


class AlphaFold3Scheduler(SchedulerMixin):
    r"""
    AF3-style coordinate diffusion scheduler.

    Implements the schedule & updates used by your SampleDiffusion:
        t̂ = c_{τ-1} * (γ + 1)
        ζ ∼ λ * sqrt(max(t̂^2 - c_{τ-1}^2, 0)) * N(0, I)
        x_{τ} = x_noisy + η * (c_τ - t̂) * ((x_{τ-1} - x̂) / t̂)

    Where x̂ is the model "denoised" prediction given x_noisy and t̂.

    Args:
        schedule: Tensor of shape (T+1,) with c_0,...,c_T (float32)
        gamma0: γ_0 in the paper (default 0.8)
        gamma_min: threshold for using γ_0 (if c_τ > gamma_min → γ=γ_0 else 0)
        noise_scale: λ
        step_scale: η
    """

    def __init__(
        self,
        schedule: Tensor,
        gamma0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
    ):
        super().__init__()

        schedule = torch.as_tensor(schedule, dtype=torch.float32)
        assert schedule.ndim == 1 and schedule.numel() >= 2, (
            "Schedule must be 1-D with length >= 2"
        )

        # Save hyperparams so `save_pretrained` / `from_pretrained` work.
        self.register_to_config(
            gamma0=gamma0,
            gamma_min=gamma_min,
            noise_scale=noise_scale,
            step_scale=step_scale,
        )
        # Buffers move with .to(device)
        self.register_buffer("c", schedule.clone(), persistent=True)

    @property
    def num_inference_steps(self) -> int:
        return int(self.c.numel() - 1)

    # ---------- Teacher-forced training helper ----------
    def add_noise(
        self, x_clean: Tensor, t_index: Tensor | int
    ) -> tuple[Tensor, Tensor]:
        """
        Build x_noisy and t_hat from clean x at schedule index `t_index` (>=1).

        Shapes:
            x_clean: (B, N, 3)
            t_index: () or (B,) integer indices in [1, T]
        Returns:
            x_noisy: (B, N, 3)
            t_hat  : (B,)      (broadcastable to (B,1,1))
        """
        device, dtype = x_clean.device, x_clean.dtype
        c = self.c.to(device=device, dtype=dtype)

        if isinstance(t_index, int):
            t_index = torch.full(
                (x_clean.shape[0],), t_index, device=device, dtype=torch.long
            )

        # c_{τ} and c_{τ-1}
        c_tau = c[t_index]  # (B,)
        c_prev = c[t_index - 1]  # (B,)

        gamma = torch.where(
            c_tau > self.config.gamma_min,
            torch.as_tensor(self.config.gamma0, device=device, dtype=dtype),
            torch.zeros((), device=device, dtype=dtype),
        )  # (B,)
        t_hat = c_prev * (gamma + 1.0)  # (B,)

        variance = (t_hat**2 - c_prev**2).clamp_min(0.0).view(-1, 1, 1)  # (B,1,1)
        eps = torch.randn_like(x_clean)
        zeta = self.config.noise_scale * variance.sqrt() * eps
        x_noisy = x_clean + zeta
        return x_noisy, t_hat  # (B,N,3), (B,)

    # ---------- Sampling step ----------
    @torch.no_grad()
    def step(
        self,
        x_denoised: Tensor,  # model's prediction x̂ (B,N,3)
        x_noisy: Tensor,  # current noisy sample (B,N,3)
        x_prev_ref: Tensor,  # previous sample x_{τ-1} (B,N,3)
        t_index: int,  # integer τ in [1, T]
    ) -> AF3SchedulerOutput:
        """
        One sampler update:
            delta = (x_{τ-1} - x̂) / t̂
            x_τ   = x_noisy + η * (c_τ - t̂) * delta
        """
        device, dtype = x_noisy.device, x_noisy.dtype
        c = self.c.to(device=device, dtype=dtype)
        c_tau = c[t_index]
        c_prev = c[t_index - 1]

        gamma = self.config.gamma0 if float(c_tau) > self.config.gamma_min else 0.0
        t_hat = c_prev * (gamma + 1.0)

        delta = (x_prev_ref - x_denoised) / t_hat
        dt = c_tau - t_hat
        x_next = x_noisy + self.config.step_scale * dt * delta
        return AF3SchedulerOutput(prev_sample=x_next)
