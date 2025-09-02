import torch.nn as nn
from torch import Tensor


class AlphaFold3TemplateEmbedder(nn.Module):
    r"""
    Template Embedder for AlphaFold 3.

    This module processes template structural information and adds it to the pair
    representation. Templates provide structural constraints from homologous
    structures that guide the prediction.

    Parameters
    ----------
    c_z : int, default=128
        Pair representation dimension
    c_template : int, default=64
        Template feature dimension
    n_head : int, default=4
        Number of attention heads

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import AlphaFold3TemplateEmbedder
    >>> batch_size, n_tokens = 2, 64
    >>> module = AlphaFold3TemplateEmbedder()
    >>> f_star = {'template_features': torch.randn(batch_size, n_tokens, n_tokens, 64)}
    >>> z_ij = torch.randn(batch_size, n_tokens, n_tokens, 128)
    >>> output = module(f_star, z_ij)
    >>> output.shape
    torch.Size([2, 64, 64, 128])
    """

    def __init__(
        self,
        c_z: int = 128,
        c_template: int = 64,
        n_head: int = 4,
    ):
        super().__init__()

        self.c_z = c_z
        self.c_template = c_template
        self.n_head = n_head

        # Template processing layers
        self.template_proj = nn.Linear(c_template, c_z, bias=False)
        self.layer_norm = nn.LayerNorm(c_z)

        # Attention mechanism for template integration
        self.attention = nn.MultiheadAttention(
            embed_dim=c_z,
            num_heads=n_head,
            batch_first=True,
        )

        # Final projection
        self.output_proj = nn.Linear(c_z, c_z, bias=False)

    def forward(self, f_star: dict, z_ij: Tensor) -> Tensor:
        r"""
        Forward pass of Template Embedder.

        Parameters
        ----------
        f_star : dict
            Dictionary containing template features with key 'template_features'
        z_ij : Tensor, shape=(batch_size, n_tokens, n_tokens, c_z)
            Current pair representations

        Returns
        -------
        z_ij : Tensor, shape=(batch_size, n_tokens, n_tokens, c_z)
            Updated pair representations with template information
        """
        # Extract template features (if available)
        if "template_features" not in f_star:
            # No templates available, return unchanged
            return z_ij

        template_features = f_star[
            "template_features"
        ]  # (batch, n_tokens, n_tokens, c_template)
        batch_size, n_tokens, _, c_template = template_features.shape

        # Project template features to pair dimension
        template_proj = self.template_proj(
            template_features
        )  # (batch, n_tokens, n_tokens, c_z)

        # Reshape for attention: (batch, n_tokens^2, c_z)
        template_flat = template_proj.reshape(batch_size, n_tokens * n_tokens, self.c_z)
        z_flat = z_ij.reshape(batch_size, n_tokens * n_tokens, self.c_z)

        # Apply layer norm
        template_flat = self.layer_norm(template_flat)

        # Self-attention to integrate template information
        template_attended, _ = self.attention(
            template_flat, template_flat, template_flat
        )

        # Add template information to pair representations
        z_updated = z_flat + template_attended

        # Final projection and reshape back
        z_updated = self.output_proj(z_updated)
        z_updated = z_updated.reshape(batch_size, n_tokens, n_tokens, self.c_z)

        return z_updated
