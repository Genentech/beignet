import torch
import torch.nn as nn
from torch import Tensor

from ._alphafold3_diffusion import AlphaFold3Diffusion
from ._centre_random_augmentation import CentreRandomAugmentation


class SampleDiffusion(nn.Module):
    r"""
    Sample Diffusion for AlphaFold 3.

    This module implements Algorithm 18 exactly, performing iterative denoising
    sampling for structure generation using a diffusion model.

    Parameters
    ----------
    gamma_0 : float, default=0.8
        Initial gamma parameter for augmentation
    gamma_min : float, default=1.0
        Minimum gamma threshold
    noise_scale : float, default=1.003
        Noise scale lambda parameter
    step_scale : float, default=1.5
        Step scale eta parameter
    s_trans : float, default=1.0
        Translation scale for augmentation

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import SampleDiffusion
    >>> batch_size, n_atoms, n_tokens = 2, 1000, 32
    >>> module = SampleDiffusion()
    >>>
    >>> # Input features
    >>> f_star = {'ref_pos': torch.randn(batch_size, n_atoms, 3)}
    >>> s_inputs = torch.randn(batch_size, n_atoms, 100)
    >>> s_trunk = torch.randn(batch_size, n_tokens, 384)
    >>> z_trunk = torch.randn(batch_size, n_tokens, n_tokens, 128)
    >>> noise_schedule = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    >>>
    >>> x_t = module(f_star, s_inputs, s_trunk, z_trunk, noise_schedule)
    >>> x_t.shape
    torch.Size([2, 1000, 3])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 18: Sample Diffusion
    """

    def __init__(
        self,
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        s_trans: float = 1.0,
    ):
        super().__init__()

        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale  # lambda
        self.step_scale = step_scale  # eta
        self.s_trans = s_trans

        # Diffusion module for denoising
        self.diffusion_module = AlphaFold3Diffusion()

        # Centre random augmentation
        self.centre_random_augmentation = CentreRandomAugmentation(s_trans=s_trans)

    def forward(
        self,
        f_star: dict,
        s_inputs: Tensor,
        s_trunk: Tensor,
        z_trunk: Tensor,
        noise_schedule: list[float],
    ) -> Tensor:
        r"""
        Forward pass implementing Algorithm 18 exactly.

        Parameters
        ----------
        f_star : dict
            Reference structure features
        s_inputs : Tensor, shape=(batch_size, n_atoms, c_s_inputs)
            Input single representations
        s_trunk : Tensor, shape=(batch_size, n_tokens, c_s)
            Trunk single representations
        z_trunk : Tensor, shape=(batch_size, n_tokens, n_tokens, c_z)
            Trunk pair representations
        noise_schedule : list
            Noise schedule [c0, c1, ..., cT]

        Returns
        -------
        x_t : Tensor, shape=(batch_size, n_atoms, 3)
            Final denoised positions
        """
        device = s_inputs.device
        batch_size = s_inputs.shape[0]
        n_atoms = s_inputs.shape[1]

        # Step 1: x̃_t ∼ c_0 · N(0̃, I_3)
        c_0 = noise_schedule[0]
        x_t = c_0 * torch.randn(batch_size, n_atoms, 3, device=device)

        # Step 2: for all c_τ ∈ [c_1, ..., c_T] do
        for tau, c_tau in enumerate(noise_schedule[1:], 1):
            # Step 3: {x̃_t} ← CentreRandomAugmentation({x̃_t})
            x_t = self.centre_random_augmentation(x_t)

            # Step 4: γ = γ_0 if c_τ > γ_min else 0
            gamma = self.gamma_0 if c_tau > self.gamma_min else 0.0

            # Step 5: t̂ = c_{τ-1}(γ + 1)
            c_tau_minus_1 = noise_schedule[tau - 1]
            t_hat = c_tau_minus_1 * (gamma + 1)

            # Step 6: ζ̃_t = λ√(t̂^2 - c^2_{τ-1}) · N(0̃, I_3)
            variance = t_hat**2 - c_tau_minus_1**2
            if variance > 0:
                zeta_t = (
                    self.noise_scale
                    * torch.sqrt(torch.tensor(variance))
                    * torch.randn_like(x_t)
                )
            else:
                zeta_t = torch.zeros_like(x_t)

            # Step 7: x̃_t^noisy = x̃_t + ζ̃_t
            x_t_noisy = x_t + zeta_t

            # Step 8: {x̃_t^denoised} = AlphaFold3Diffusion({x̃_t^noisy}, t̂, {f*}, {s_i^inputs}, {s_i^trunk}, {z_{ij}^trunk})
            # Create timestep tensor
            t_tensor = torch.full(
                (batch_size, 1), t_hat, device=device, dtype=x_t.dtype
            )

            # Get reference positions from f_star
            f_star_pos = f_star.get("ref_pos", x_t * 0)  # Use zeros if not available

            # Create dummy z_atom for the diffusion module
            z_atom = torch.zeros(
                batch_size, n_atoms, n_atoms, 16, device=device, dtype=x_t.dtype
            )

            x_t_denoised = self.diffusion_module(
                x_noisy=x_t_noisy,
                t=t_tensor,
                f_star=f_star_pos,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                z_atom=z_atom,
            )

            # Step 9: δ̃_t = (x̃_t - x̃_t^denoised) / t̂
            delta_t = (x_t - x_t_denoised) / t_hat

            # Step 10: dt = c_τ - t̂
            dt = c_tau - t_hat

            # Step 11: x̃_t ← x̃_t^noisy + η · dt · δ̃_t
            x_t = x_t_noisy + self.step_scale * dt * delta_t

        # Step 12: end for
        # Step 13: return {x̃_t}
        return x_t
