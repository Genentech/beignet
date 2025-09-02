import torch
import torch.nn as nn
from torch import Tensor

from ._alphafold3_confidence import AlphaFold3Confidence
from ._alphafold3_distogram import AlphaFold3Distogram
from ._alphafold3_msa import AlphaFold3MSA
from ._alphafold3_template_embedder import AlphaFold3TemplateEmbedder
from ._input_feature_embedder import InputFeatureEmbedder
from ._pairformer_stack import PairformerStack
from ._relative_position_encoding import RelativePositionEncoding
from ._sample_diffusion import SampleDiffusion


class AlphaFold3(nn.Module):
    r"""
    Main Inference Loop for AlphaFold 3.

    This module implements Algorithm 1 exactly, which is the main inference
    pipeline for AlphaFold 3. It processes input features through multiple
    stages including feature embedding, MSA processing, template embedding,
    Pairformer stacks, diffusion sampling, and confidence prediction.

    Parameters
    ----------
    n_cycle : int, default=4
        Number of recycling cycles
    c_s : int, default=384
        Single representation dimension
    c_z : int, default=128
        Pair representation dimension
    c_m : int, default=64
        MSA representation dimension
    c_template : int, default=64
        Template feature dimension
    n_blocks_pairformer : int, default=48
        Number of blocks in PairformerStack
    n_head : int, default=16
        Number of attention heads

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import AlphaFold3
    >>> batch_size, n_tokens = 2, 64
    >>> module = AlphaFold3(n_cycle=2)  # Smaller for example
    >>> f_star = {
    ...     'asym_id': torch.randint(0, 5, (batch_size, n_tokens)),
    ...     'residue_index': torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
    ...     'entity_id': torch.randint(0, 3, (batch_size, n_tokens)),
    ...     'token_index': torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
    ...     'sym_id': torch.randint(0, 10, (batch_size, n_tokens)),
    ...     'token_bonds': torch.randn(batch_size, n_tokens, n_tokens, 32)
    ... }
    >>> outputs = module(f_star)
    >>> outputs['x_pred'].shape
    torch.Size([2, 64, 3])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 1: Main Inference Loop
    """

    def __init__(
        self,
        n_cycle: int = 4,
        c_s: int = 384,
        c_z: int = 128,
        c_m: int = 64,
        c_template: int = 64,
        n_blocks_pairformer: int = 48,
        n_head: int = 16,
    ):
        super().__init__()

        self.n_cycle = n_cycle
        self.c_s = c_s
        self.c_z = c_z

        # Step 1: Input Feature Embedder
        self.input_feature_embedder = InputFeatureEmbedder(
            c_atom=128,
            c_atompair=16,
            c_token=c_s,
        )

        # Step 2-3: Linear projections for initial representations
        self.single_linear = nn.Linear(c_s, c_s, bias=False)  # s_i^init
        self.pair_linear_i = nn.Linear(c_s, c_z, bias=False)  # z_ij^init from s_i
        self.pair_linear_j = nn.Linear(c_s, c_z, bias=False)  # z_ij^init from s_j

        # Step 4: Relative Position Encoding
        self.relative_position_encoding = RelativePositionEncoding(
            c_z=c_z,
        )

        # Step 5: Token bonds projection
        self.token_bonds_linear = nn.Linear(
            32, c_z, bias=False
        )  # Assuming 32 bond features

        # Step 8: Layer norm for pair initialization
        self.pair_layer_norm = nn.LayerNorm(c_z)

        # Step 9: Template Embedder
        self.template_embedder = AlphaFold3TemplateEmbedder(
            c_z=c_z,
            c_template=c_template,
        )

        # Step 10: MSA Module
        self.msa_module = AlphaFold3MSA(
            c_m=c_m,
            c_z=c_z,
            c_s=c_s,
        )

        # Step 11: Single representation update
        self.single_update_linear = nn.Linear(c_s, c_s, bias=False)
        self.single_layer_norm = nn.LayerNorm(c_s)

        # Step 12: Pairformer Stack
        self.pairformer_stack = PairformerStack(
            n_block=n_blocks_pairformer,
            c_s=c_s,
            c_z=c_z,
            n_head_single=n_head,
            n_head_pair=n_head // 4,  # Typically fewer heads for pair attention
        )

        # Step 15: Sample Diffusion
        self.sample_diffusion = SampleDiffusion()

        # Step 16: Confidence Head
        self.confidence_head = AlphaFold3Confidence(
            c_s=c_s,
            c_z=c_z,
        )

        # Step 17: Distogram Head
        self.distogram_head = AlphaFold3Distogram(
            c_z=c_z,
        )

    def forward(self, f_star: dict[str, Tensor]) -> dict[str, Tensor]:
        r"""
        Forward pass implementing Algorithm 1 exactly.

        Parameters
        ----------
        f_star : dict
            Dictionary containing input features with keys:
            - 'asym_id': asymmetric unit IDs (batch, n_tokens)
            - 'residue_index': residue indices (batch, n_tokens)
            - 'entity_id': entity IDs (batch, n_tokens)
            - 'token_index': token indices (batch, n_tokens)
            - 'sym_id': symmetry IDs (batch, n_tokens)
            - 'token_bonds': token bond features (batch, n_tokens, n_tokens, bond_dim)
            - Optional: 'template_features', 'msa_features', etc.

        Returns
        -------
        outputs : dict
            Dictionary containing:
            - 'x_pred': predicted coordinates (batch, n_tokens, 3)
            - 'p_plddt': pLDDT confidence (batch, n_tokens)
            - 'p_pae': PAE confidence (batch, n_tokens, n_tokens)
            - 'p_pde': PDE confidence (batch, n_tokens, n_tokens)
            - 'p_resolved': resolved confidence (batch, n_tokens)
            - 'p_distogram': distance distributions (batch, n_tokens, n_tokens, n_bins)
        """
        # Step 1: Input Feature Embedder
        embeddings = self.input_feature_embedder(f_star)
        s_inputs = embeddings["single"]  # (batch, n_tokens, c_s)

        # Step 2: Initialize single representation
        s_i_init = self.single_linear(s_inputs)  # (batch, n_tokens, c_s)

        # Step 3: Initialize pair representation
        # z_ij^init = LinearNoBias(s_i^inputs) + LinearNoBias(s_j^inputs)
        pair_i = self.pair_linear_i(s_inputs).unsqueeze(-2)  # (batch, n_tokens, 1, c_z)
        pair_j = self.pair_linear_j(s_inputs).unsqueeze(-3)  # (batch, 1, n_tokens, c_z)
        z_ij_init = pair_i + pair_j  # (batch, n_tokens, n_tokens, c_z)

        # Step 4: Add relative position encoding
        z_ij_init = z_ij_init + self.relative_position_encoding(f_star)

        # Step 5: Add token bonds (if available)
        if "token_bonds" in f_star:
            token_bonds = f_star["token_bonds"]  # (batch, n_tokens, n_tokens, bond_dim)
            z_ij_init = z_ij_init + self.token_bonds_linear(token_bonds)

        # Step 6: Initialize accumulators
        z_ij = torch.zeros_like(z_ij_init)
        s_i = torch.zeros_like(s_i_init)

        # Step 7-14: Main recycling loop
        for c in range(self.n_cycle):
            # Step 8: Update pair representation
            z_ij = z_ij_init + self.pair_layer_norm(z_ij)

            # Step 9: Template Embedder
            z_ij = z_ij + self.template_embedder(f_star, z_ij)

            # Step 10: MSA Module
            if "msa_features" in f_star:
                z_ij = z_ij + self.msa_module(
                    f_star["msa_features"],
                    f_star.get("has_deletion"),
                    f_star.get("deletion_value"),
                    s_inputs,
                    z_ij,
                )

            # Step 11: Update single representation
            s_i = s_i_init + self.single_update_linear(self.single_layer_norm(s_i))

            # Step 12: Pairformer Stack
            s_i, z_ij = self.pairformer_stack(s_i, z_ij)

            # Step 13: Copy for next iteration (handled by loop)

        # Step 15: Sample Diffusion
        x_pred = self.sample_diffusion(
            f_star, s_inputs, s_i, z_ij, noise_schedule=torch.linspace(1.0, 0.01, 20)
        )

        # Step 16: Confidence Head
        confidence_outputs = self.confidence_head(
            {"token_single_initial_repr": s_inputs}, s_i, z_ij, x_pred
        )

        # Step 17: Distogram Head
        p_distogram = self.distogram_head(z_ij)

        # Step 18: Return all outputs
        return {
            "x_pred": x_pred,
            "p_plddt": confidence_outputs["p_plddt"],
            "p_pae": confidence_outputs["p_pae"],
            "p_pde": confidence_outputs["p_pde"],
            "p_resolved": confidence_outputs["p_resolved"],
            "p_distogram": p_distogram,
        }
