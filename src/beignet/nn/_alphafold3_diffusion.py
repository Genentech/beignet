import torch
import torch.nn as nn
from torch import Tensor

from . import FourierEmbedding, RelativePositionEncoding, Transition
from ._atom_attention_decoder import AtomAttentionDecoder
from ._atom_attention_encoder import AtomAttentionEncoder
from ._diffusion_transformer import DiffusionTransformer


class AlphaFold3Diffusion(nn.Module):
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

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import AlphaFold3Diffusion
    >>> batch_size, n_tokens, n_atoms = 2, 32, 1000
    >>> module = AlphaFold3Diffusion()
    >>>
    >>> # Input data
    >>> x_noisy = torch.randn(batch_size, n_atoms, 3)  # Noisy positions
    >>> t = torch.randn(batch_size, 1)  # Timesteps
    >>> f_star = torch.randn(batch_size, n_atoms, 3)  # Target positions
    >>> s_inputs = torch.randn(batch_size, n_atoms, 100)  # Input single
    >>> s_trunk = torch.randn(batch_size, n_tokens, 384)  # Trunk single
    >>> z_trunk = torch.randn(batch_size, n_tokens, n_tokens, 128)  # Trunk pair
    >>> z_atom = torch.randn(batch_size, n_atoms, n_atoms, 16)  # Atom pairs
    >>>
    >>> x_out = module(x_noisy, t, f_star, s_inputs, s_trunk, z_trunk, z_atom)
    >>> x_out.shape
    torch.Size([2, 1000, 3])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 20: Diffusion Module
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
        self.diffusion_conditioning = DiffusionConditioning(
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
        self.atom_attention_decoder = AtomAttentionDecoder(
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


class DiffusionConditioning(nn.Module):
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

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import DiffusionConditioning
    >>> batch_size, n_atoms = 2, 100
    >>> module = DiffusionConditioning()
    >>>
    >>> # Input data
    >>> t = torch.randn(batch_size, 1)  # Timesteps
    >>> f_star = torch.randn(batch_size, n_atoms, 3)  # Target positions
    >>> s_trunk = torch.randn(batch_size, n_atoms, 384)  # Trunk single
    >>> s_inputs = torch.randn(batch_size, n_atoms, 100)  # Input single
    >>> z_trunk = torch.randn(batch_size, n_atoms, n_atoms, 128)  # Trunk pair
    >>>
    >>> s, z = module(t, f_star, s_inputs, s_trunk, z_trunk)
    >>> s.shape, z.shape
    (torch.Size([2, 100, 384]), torch.Size([2, 100, 100, 128]))

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 21: Diffusion Conditioning
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
        self.fourier_embedding = FourierEmbedding(c=256)

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
