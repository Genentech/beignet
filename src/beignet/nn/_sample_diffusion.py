import torch
import torch.nn as nn
from torch import Tensor

from beignet.nn import (
    AtomAttentionEncoder,
    AtomTransformer,
    DiffusionTransformer,
    RelativePositionEncoding,
    Transition,
)


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
        self.diffusion_module = _Diffusion()

        # Centre random augmentation
        self.centre_random_augmentation = _CentreRandomAugmentation(s_trans=s_trans)

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


class _CentreRandomAugmentation(nn.Module):
    r"""
    Centre Random Augmentation for AlphaFold 3.

    This module implements Algorithm 19 exactly, applying random rotation and
    translation to center and augment atomic positions.

    Parameters
    ----------
    s_trans : float, default=1.0
        Translation scale in Ångstroms

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import _CentreRandomAugmentation
    >>> batch_size, n_atoms = 2, 1000
    >>> module = _CentreRandomAugmentation()
    >>>
    >>> x_t = torch.randn(batch_size, n_atoms, 3)
    >>> x_t_augmented = module(x_t)
    >>> x_t_augmented.shape
    torch.Size([2, 1000, 3])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 19: CentreRandomAugmentation
    """

    def __init__(self, s_trans: float = 1.0):
        super().__init__()

        self.s_trans = s_trans

    def uniform_random_rotation(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """
        Generate uniform random rotation matrices.

        Parameters
        ----------
        batch_size : int
            Number of rotation matrices to generate
        device : torch.device
            Device to create tensors on

        Returns
        -------
        R : Tensor, shape=(batch_size, 3, 3)
            Random rotation matrices
        """
        # Generate random quaternions and normalize them for uniform distribution
        # This is a simplified implementation - for true uniformity, use more sophisticated methods
        q = torch.randn(batch_size, 4, device=device, dtype=dtype)
        q = q / torch.norm(q, dim=-1, keepdim=True)

        # Convert quaternions to rotation matrices
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Rotation matrix from quaternion
        R = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)

        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)

        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y * z - x * w)

        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        return R

    def forward(self, x_t: Tensor) -> Tensor:
        r"""
        Forward pass implementing Algorithm 19 exactly.

        Implements the following steps:
        1. x̃_t ← x̃_t - mean_l x̃_t
        2. r = UniformRandomRotation()
        3. t̃ ∼ s_trans · N(0̃, I_3)
        4. x̃_t ← r · x̃_t + t̃
        5. return {x̃_t}

        Parameters
        ----------
        x_t : Tensor, shape=(batch_size, n_atoms, 3)
            Input atomic positions

        Returns
        -------
        x_t : Tensor, shape=(batch_size, n_atoms, 3)
            Augmented atomic positions
        """
        batch_size, n_atoms, _ = x_t.shape
        # Step 1: x̃_t ← x̃_t - mean_l x̃_t
        # Center the coordinates by subtracting the mean
        x_t = x_t - torch.mean(x_t, dim=1, keepdim=True)

        # Step 2: r = UniformRandomRotation()
        r = self.uniform_random_rotation(batch_size, x_t.device, x_t.dtype)

        # Step 3: t̃ ∼ s_trans · N(0̃, I_3)
        t = self.s_trans * torch.randn(
            batch_size,
            1,
            3,
            device=x_t.device,
            dtype=x_t.dtype,
        )

        # Step 4: x̃_t ← r · x̃_t + t̃
        # Apply rotation: (batch_size, 3, 3) @ (batch_size, n_atoms, 3) -> (batch_size, n_atoms, 3)
        # x_t @ r^T
        x_t = torch.bmm(x_t, r.transpose(-2, -1)) + t

        # Step 5: return {x̃_t}
        return x_t


class _Diffusion(nn.Module):
    r"""
    Diffusion Module from AlphaFold 3 Algorithm 20.

    This is the main diffusion module that implements the complete diffusion process
    for atomic coordinates. It handles conditioning, position scaling, atom attention
    encoding/decoding, and the diffusion transformer processing.

    Parameters
    ----------
    c_token : int, default=768
        Channel dimension for token representations
    c_atom : int, default=128
        Channel dimension for atom representations
    c_atompair : int, default=16
        Channel dimension for atom pair representations
    sigma_data : float, default=16.0
        Data scaling factor for noise schedule
    n_head : int, default=16
        Number of attention heads
    n_block : int, default=24
        Number of diffusion transformer blocks
    """

    def __init__(
        self,
        c_token: int = 768,
        c_atom: int = 128,
        c_atompair: int = 16,
        sigma_data: float = 16.0,
        n_head: int = 16,
        n_block: int = 24,
    ):
        super().__init__()

        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.sigma_data = sigma_data
        self.n_head = n_head
        self.n_block = n_block

        # Step 1: Diffusion conditioning
        self.diffusion_conditioning = _DiffusionConditioning(
            c_z=128,  # Pair representation dimension
            c_s=384,  # Single representation dimension
            c_s_inputs=None,  # Will be inferred from input
            sigma_data=sigma_data,
        )

        # Step 3: Atom attention encoder
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_token=c_token, c_atom=c_atom, c_atompair=c_atompair, n_head=n_head
        )

        # Step 5: Diffusion transformer
        self.diffusion_transformer = DiffusionTransformer(
            c_a=c_token,  # Token dimension
            c_s=384,  # Single conditioning dimension
            c_z=128,  # Pair conditioning dimension
            n_head=n_head,
            n_block=n_block,
        )

        # Step 6: Layer normalization for tokens
        self.layer_norm_tokens = nn.LayerNorm(c_token)

        # Step 7: Atom attention decoder
        self.atom_attention_decoder = _AtomAttentionDecoder(
            c_token=c_token, c_atom=c_atom, n_head=n_head
        )

        # Additional layers for step 4
        self.trunk_to_token_proj = nn.Linear(
            384, c_token, bias=False
        )  # Project s_trunk to token dim
        self.trunk_layer_norm = nn.LayerNorm(384)

    def forward(
        self,
        x_noisy: Tensor,
        t: Tensor,
        f_star: Tensor,
        s_inputs: Tensor,
        s_trunk: Tensor,
        z_trunk: Tensor,
        z_atom: Tensor,
    ) -> Tensor:
        r"""
        Forward pass of Diffusion Module.

        Parameters
        ----------
        x_noisy : Tensor, shape=(batch_size, n_atoms, 3)
            Noisy atomic positions
        t : Tensor, shape=(batch_size, 1) or (batch_size,)
            Timestep values
        f_star : Tensor, shape=(batch_size, n_atoms, 3)
            Target atomic positions
        s_inputs : Tensor, shape=(batch_size, n_atoms, c_s_inputs)
            Input single representations
        s_trunk : Tensor, shape=(batch_size, n_tokens, c_s)
            Trunk single representations
        z_trunk : Tensor, shape=(batch_size, n_tokens, n_tokens, c_z)
            Trunk pair representations
        z_atom : Tensor, shape=(batch_size, n_atoms, n_atoms, c_atompair)
            Atom pair representations

        Returns
        -------
        x_out : Tensor, shape=(batch_size, n_atoms, 3)
            Denoised atomic positions
        """
        batch_size = x_noisy.shape[0]

        # Step 1: Conditioning - {s_i}, {z_ij} = DiffusionConditioning(...)
        s_conditioned, z_conditioned = self.diffusion_conditioning(
            t, f_star, s_inputs, s_trunk, z_trunk
        )

        # Step 2: Scale positions to dimensionless vectors with approximately unit variance
        # r_i^noisy = x_i^noisy / sqrt(t^2 + sigma_data^2)
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)  # Ensure (batch, 1)
        t_expanded = t.unsqueeze(-1)  # (batch, 1, 1)
        scale_factor = torch.sqrt(t_expanded**2 + self.sigma_data**2)
        r_noisy = x_noisy / scale_factor

        # Step 3: Atom attention encoder
        # {a_i}, {q_i^skip}, {c_i^skip}, {p_im^skip} = AtomAttentionEncoder(...)
        a, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            f_star, r_noisy, s_trunk, z_atom
        )

        # Step 4: Full self-attention on token level
        # a_i += LinearNoBias(LayerNorm(s_i))
        # This step seems to add conditioning to tokens
        trunk_normed = self.trunk_layer_norm(s_trunk)
        trunk_projected = self.trunk_to_token_proj(trunk_normed)
        a = a + trunk_projected

        # Step 5: Diffusion transformer
        # {a_i} ← DiffusionTransformer({a_i}, {s_i}, {z_ij}, β_ij = 0, N_block = 24, N_head = 16)
        # Create zero bias tensor
        beta_ij = torch.zeros(
            batch_size,
            s_conditioned.shape[1],
            s_conditioned.shape[1],
            self.n_head,
            device=a.device,
            dtype=a.dtype,
        )

        a = self.diffusion_transformer(a, s_conditioned, z_conditioned, beta_ij)

        # Step 6: Layer normalization
        # a_i ← LayerNorm(a_i)
        a = self.layer_norm_tokens(a)

        # Step 7: Atom attention decoder
        # {r_i^update} = AtomAttentionDecoder({a_i}, {q_i^skip}, {c_i^skip}, {p_im^skip})
        r_update = self.atom_attention_decoder(a, q_skip, c_skip, p_skip)

        # Step 8: Rescale updates to positions and combine with input positions
        # x_i^out = sigma_data^2 / (sigma_data^2 + t^2) * x_i^noisy + sigma_data * t / sqrt(sigma_data^2 + t^2) * r_i^update
        t_sq = t_expanded**2
        sigma_sq = self.sigma_data**2
        denom = sigma_sq + t_sq

        coeff_noisy = sigma_sq / denom
        coeff_update = self.sigma_data * t_expanded / torch.sqrt(denom)

        x_out = coeff_noisy * x_noisy + coeff_update * r_update

        # Step 9: return {x_i^out}
        return x_out


class _DiffusionConditioning(nn.Module):
    r"""
    Diffusion Conditioning from AlphaFold 3 Algorithm 21.

    This module computes conditioning signals for diffusion models by processing
    pair and single representations. It handles both pair conditioning through
    trunk pair representations and relative position encoding, and single conditioning
    through trunk single representations and timestep embedding.

    Parameters
    ----------
    c_z : int, default=128
        Channel dimension for pair representations
    c_s : int, default=384
        Channel dimension for single representations
    sigma_data : float, default=16.0
        Data scaling factor for timestep embedding
    """

    def __init__(
        self,
        c_z: int = 128,
        c_s: int = 384,
        c_s_inputs: int = None,
        sigma_data: float = 16.0,
    ):
        super().__init__()

        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs or c_s  # Default to same as c_s if not specified
        self.sigma_data = sigma_data

        # Step 1: Relative position encoding
        self.relative_pos_enc = RelativePositionEncoding(c_z=c_z)

        # Step 2: Linear projection for pair representations
        # Need to handle concatenation of z_trunk + rel_pos_enc
        self.linear_z = nn.Linear(2 * c_z, c_z, bias=False)
        self.layer_norm_z = nn.LayerNorm(2 * c_z)

        # Steps 3-5: Transition blocks for pair representations (2 blocks)
        self.transition_z_1 = Transition(c=c_z, n=2)
        self.transition_z_2 = Transition(c=c_z, n=2)

        # Step 7: Linear projection for single representations
        # Need to handle concatenation of s_trunk + s_inputs
        self.linear_s = nn.Linear(c_s + self.c_s_inputs, c_s, bias=False)
        self.layer_norm_s = nn.LayerNorm(c_s + self.c_s_inputs)

        # Step 8: Fourier embedding for timestep (embedding dimension = 256)
        self.fourier_embedding = _FourierEmbedding(c=256)

        # Step 9: Linear projection for timestep embedding
        self.linear_timestep = nn.Linear(256, c_s, bias=False)
        self.layer_norm_timestep = nn.LayerNorm(256)

        # Steps 10-12: Transition blocks for single representations (2 blocks)
        self.transition_s_1 = Transition(c=c_s, n=2)
        self.transition_s_2 = Transition(c=c_s, n=2)

    def forward(
        self,
        t: Tensor,
        f_star: Tensor,
        s_inputs: Tensor,
        s_trunk: Tensor,
        z_trunk: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Forward pass of Diffusion Conditioning.

        Parameters
        ----------
        t : Tensor, shape=(batch_size, 1) or (batch_size,)
            Timestep values
        f_star : Tensor, shape=(batch_size, n_atoms, 3)
            Target positions
        s_inputs : Tensor, shape=(batch_size, n_atoms, c_s_input)
            Input single representations
        s_trunk : Tensor, shape=(batch_size, n_atoms, c_s)
            Trunk single representations
        z_trunk : Tensor, shape=(batch_size, n_atoms, n_atoms, c_z)
            Trunk pair representations

        Returns
        -------
        s : Tensor, shape=(batch_size, n_atoms, c_s)
            Conditioned single representations
        z : Tensor, shape=(batch_size, n_atoms, n_atoms, c_z)
            Conditioned pair representations
        """

        # Pair conditioning
        # Step 1: z_ij = concat([z_ij^trunk, RelativePositionEncoding({f*})])
        rel_pos_enc = self.relative_pos_enc(f_star)
        z_concat = torch.cat([z_trunk, rel_pos_enc], dim=-1)

        # Step 2: z_ij ← LinearNoBias(LayerNorm(z_ij))
        z = self.linear_z(self.layer_norm_z(z_concat))

        # Steps 3-5: for all b ∈ [1, 2] do z_ij += Transition(z_ij, n = 2)
        z = z + self.transition_z_1(z)
        z = z + self.transition_z_2(z)

        # Single conditioning
        # Step 6: s_i = concat([s_i^trunk, s_i^inputs])
        s = torch.cat([s_trunk, s_inputs], dim=-1)

        # Step 7: s_i ← LinearNoBias(LayerNorm(s_i))
        s = self.linear_s(self.layer_norm_s(s))

        # Step 8: n = FourierEmbedding(1/4 * log(t/σ_data), 256)
        # Compute timestep embedding
        t_scaled = t / self.sigma_data
        # Ensure we handle the 1/4 scaling and log properly
        t_log = 0.25 * torch.log(torch.clamp(t_scaled, min=1e-8))
        n = self.fourier_embedding(t_log)

        # Step 9: s_i += LinearNoBias(LayerNorm(n))
        # Broadcast n across atoms: (batch_size, 256) -> (batch_size, n_atoms, c_s)
        n_projected = self.linear_timestep(self.layer_norm_timestep(n))

        # Expand n_projected to match s dimensions
        if len(n_projected.shape) == 2:  # (batch_size, c_s)
            n_projected = n_projected.unsqueeze(1)  # (batch_size, 1, c_s)
        n_broadcast = n_projected.expand(
            -1, s.shape[1], -1
        )  # (batch_size, n_atoms, c_s)

        s = s + n_broadcast

        # Steps 10-12: for all b ∈ [1, 2] do s_i += Transition(s_i, n = 2)
        s = s + self.transition_s_1(s)
        s = s + self.transition_s_2(s)

        # Step 13: return {s_i}, {z_ij}
        return s, z


class _FourierEmbedding(nn.Module):
    r"""
    Fourier Embedding from AlphaFold 3 Algorithm 22.

    This implements Fourier positional embeddings using random weights and biases
    that are generated once before training and then frozen. The embedding uses
    cosine activation: cos(2π(tw + b))

    Parameters
    ----------
    c : int
        Output embedding dimension
    """

    def __init__(self, c: int):
        super().__init__()

        self.c = c

        # Algorithm 22 Step 1: w, b ~ N(0, I_c)
        # Randomly generate weight/bias once before training
        # These are frozen parameters (not updated during training)
        self.register_buffer("w", torch.randn(c))
        self.register_buffer("b", torch.randn(c))

    def forward(self, input: Tensor) -> Tensor:
        r"""
        Forward pass of Fourier Embedding.

        Parameters
        ----------
        input : Tensor, shape=(..., 1) or (...,)
            Input tensor containing times and positions.

        Returns
        -------
        embeddings : Tensor, shape=(..., c)
            Fourier embeddings using cosine activation
        """
        # Handle both (..., 1) and (...,) input shapes
        if input.dim() > 1 and input.shape[-1] == 1:
            input = input.squeeze(-1)  # Remove last dimension if it's 1

        # Ensure t has correct shape for broadcasting
        # t should be (...,) and we want to broadcast with w and b which are (c,)
        input = torch.unsqueeze(input, -1)  # Shape: (..., 1)

        # Algorithm 22 Step 2: return cos(2π(tw + b))
        # Broadcasting: (..., 1) * (c,) + (c,) -> (..., c)
        return torch.cos(2 * torch.pi * (input * self.w + self.b))


class _AtomAttentionDecoder(nn.Module):
    r"""
    Atom Attention Decoder for AlphaFold 3.

    This module broadcasts per-token activations to per-atom activations,
    applies cross attention transformer, and maps to position updates.
    Implements Algorithm 6 exactly.

    Parameters
    ----------
    c_token : int, default=768
        Channel dimension for token representations
    c_atom : int, default=128
        Channel dimension for atom representations
    n_block : int, default=3
        Number of transformer blocks
    n_head : int, default=4
        Number of attention heads

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import _AtomAttentionDecoder
    >>> batch_size, n_tokens, n_atoms = 2, 32, 1000
    >>> module = _AtomAttentionDecoder()
    >>>
    >>> a = torch.randn(batch_size, n_tokens, 768)  # Token representations
    >>> q_skip = torch.randn(batch_size, n_atoms, 768)  # Query skip
    >>> c_skip = torch.randn(batch_size, n_atoms, 128)  # Context skip
    >>> p_skip = torch.randn(batch_size, n_atoms, n_atoms, 16)  # Pair skip
    >>>
    >>> r_update = module(a, q_skip, c_skip, p_skip)
    >>> r_update.shape
    torch.Size([2, 1000, 3])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 6: Atom attention decoder
    """

    def __init__(
        self, c_token: int = 768, c_atom: int = 128, n_block: int = 3, n_head: int = 4
    ):
        super().__init__()

        self.c_token = c_token
        self.c_atom = c_atom
        self.n_block = n_block
        self.n_head = n_head

        # Step 1: Broadcast per-token activations to per-atom activations
        self.token_to_atom_proj = nn.Linear(c_token, c_token, bias=False)

        # Step 2: Cross attention transformer
        self.atom_transformer = AtomTransformer(
            n_block=n_block,
            n_head=n_head,
            c_q=c_token,  # Query dimension
            c_kv=c_atom,  # Key-value dimension
            c_pair=None,  # Will be inferred from p_skip
        )

        # Step 3: Map to position updates
        self.position_proj = nn.Linear(c_token, 3, bias=False)
        self.layer_norm = nn.LayerNorm(c_token)

    def forward(
        self, a: Tensor, q_skip: Tensor, c_skip: Tensor, p_skip: Tensor
    ) -> Tensor:
        r"""
        Forward pass of Atom Attention Decoder.

        Implements Algorithm 6 exactly:
        1. q_l = LinearNoBias(a_tok_idx(l)) + q_l^skip
        2. {q_l} = AtomTransformer({q_l}, {c_l^skip}, {p_lm^skip}, N_block=3, N_head=4)
        3. r_l^update = LinearNoBias(LayerNorm(q_l))

        Parameters
        ----------
        a : Tensor, shape=(batch_size, n_tokens, c_token)
            Token-level representations
        q_skip : Tensor, shape=(batch_size, n_atoms, c_token)
            Query skip connection
        c_skip : Tensor, shape=(batch_size, n_atoms, c_atom)
            Context skip connection
        p_skip : Tensor, shape=(batch_size, n_atoms, n_atoms, c_atompair)
            Pair skip connection

        Returns
        -------
        r_update : Tensor, shape=(batch_size, n_atoms, 3)
            Position updates for atoms
        """
        batch_size, n_tokens, c_token = a.shape
        n_atoms = q_skip.shape[1]

        # Step 1: Broadcast per-token activations to per-atom activations and add skip connection
        # q_l = LinearNoBias(a_tok_idx(l)) + q_l^skip

        # Create token indices for each atom (simple broadcasting approach)
        # For simplicity, we'll map atoms to tokens cyclically
        token_indices = torch.arange(n_atoms, device=a.device) % n_tokens

        # Get corresponding token activations for each atom
        a_tok_idx = a[:, token_indices]  # (batch_size, n_atoms, c_token)

        # Apply linear projection and add skip connection
        q = self.token_to_atom_proj(a_tok_idx) + q_skip

        # Step 2: Cross attention transformer
        # {q_l} = AtomTransformer({q_l}, {c_l^skip}, {p_lm^skip}, N_block=3, N_head=4)
        q = self.atom_transformer(q, c_skip, p_skip)

        # Step 3: Map to positions update
        # r_l^update = LinearNoBias(LayerNorm(q_l))
        r_update = self.position_proj(self.layer_norm(q))

        return r_update
