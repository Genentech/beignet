import torch
import torch.nn as nn
from torch import Tensor

from ._fourier_embedding import FourierEmbedding
from ._relative_position_encoding import RelativePositionEncoding
from ._transition import Transition


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
