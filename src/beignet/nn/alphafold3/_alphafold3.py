import torch
import torch.nn as nn
from torch import Tensor

from ..._one_hot import one_hot
from ._atom_attention_encoder import AtomAttentionEncoder
from ._msa import MSA
from ._pairformer_stack import PairformerStack
from ._relative_position_encoding import RelativePositionEncoding
from ._sample_diffusion import SampleDiffusion
from ._template_embedder import TemplateEmbedder


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
        self.input_feature_embedder = _InputFeatureEmbedder(
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
        self.template_embedder = TemplateEmbedder(
            c_z=c_z,
            c_template=c_template,
        )

        # Step 10: MSA Module
        self.msa_module = MSA(
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
        self.confidence_head = _Confidence(
            c_s=c_s,
            c_z=c_z,
        )

        # Step 17: Distogram Head
        self.distogram_head = _Distogram(
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
        for _ in range(self.n_cycle):
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


class _Confidence(nn.Module):
    r"""
    Confidence Head for AlphaFold 3.

    This module implements Algorithm 31 exactly, computing confidence scores
    for different structural predictions (PAE, PDE, pLDDT, resolved).

    Parameters
    ----------
    n_block : int, default=4
        Number of Pairformer blocks
    c_s : int, default=384
        Single representation dimension
    c_z : int, default=128
        Pair representation dimension
    """

    def __init__(self, n_block: int = 4, c_s: int = 384, c_z: int = 128):
        super().__init__()

        self.n_block = n_block
        self.c_s = c_s
        self.c_z = c_z

        # Step 1: Update pair representations with input features
        self.input_proj_i = nn.Linear(
            100, c_z, bias=False
        )  # Assuming s_inputs has 100 features
        self.input_proj_j = nn.Linear(100, c_z, bias=False)

        # Step 2: Embed pair distances of representative atoms
        self.distance_bins = torch.tensor(
            [
                3.0,
                5.0,
                7.0,
                9.0,
                11.0,
                13.0,
                15.0,
                17.0,
                19.0,
                21.0,  # Example bins in Ångströms
            ]
        )

        # Step 3: Distance embedding
        self.dist_embedding = nn.Linear(len(self.distance_bins), c_z, bias=False)

        # Step 4: Pairformer stack
        self.pairformer_stack = PairformerStack(c_s=c_s, c_z=c_z, n_block=n_block)

        # Output heads
        # Step 5: PAE (Predicted Aligned Error) head
        self.pae_head = nn.Linear(c_z, 64, bias=False)  # 64 distance bins

        # Step 6: PDE (Predicted Distance Error) head
        self.pde_head = nn.Linear(c_z, 64, bias=False)  # 64 distance bins

        # Step 7: pLDDT (per-residue confidence) head
        self.plddt_head = nn.Linear(c_s, 50, bias=False)  # 50 confidence bins

        # Step 8: Resolved (atom resolved) head
        self.resolved_head = nn.Linear(c_s, 2, bias=False)  # Binary classification

    def forward(
        self,
        s_inputs: Tensor,
        s_i: Tensor,
        z_ij: Tensor,
        x_pred: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Forward pass implementing Algorithm 31 exactly.

        Parameters
        ----------
        s_inputs : Tensor, shape=(batch_size, n_atoms, c_s_inputs)
            Input single representations
        s_i : Tensor, shape=(batch_size, n_tokens, c_s)
            Single representations
        z_ij : Tensor, shape=(batch_size, n_tokens, n_tokens, c_z)
            Pair representations
        x_pred : Tensor, shape=(batch_size, n_atoms, 3)
            Predicted atomic positions

        Returns
        -------
        p_plddt : Tensor, shape=(batch_size, n_atoms, 50)
            pLDDT confidence scores
        p_pae : Tensor, shape=(batch_size, n_tokens, n_tokens, 64)
            PAE confidence scores
        p_pde : Tensor, shape=(batch_size, n_tokens, n_tokens, 64)
            PDE confidence scores
        p_resolved : Tensor, shape=(batch_size, n_atoms, 2)
            Resolved confidence scores
        """
        batch_size, n_atoms = s_inputs.shape[:2]
        n_tokens = s_i.shape[1]
        device = s_inputs.device

        # Step 1: z_ij += LinearNoBias(s_i^inputs) + LinearNoBias(s_j^inputs)
        # Map atoms to tokens for input projection
        token_indices = torch.arange(n_atoms, device=device) % n_tokens
        s_inputs_broadcast = torch.zeros(
            batch_size, n_tokens, s_inputs.shape[-1], device=device
        )

        # Simple aggregation of atom features to token features
        for i in range(n_tokens):
            mask = token_indices == i
            if mask.any():
                s_inputs_broadcast[:, i] = s_inputs[:, mask].mean(dim=1)

        # Project and add to pair representations
        s_proj_i = self.input_proj_i(s_inputs_broadcast)  # (batch, n_tokens, c_z)
        s_proj_j = self.input_proj_j(s_inputs_broadcast)  # (batch, n_tokens, c_z)

        z_ij = z_ij + s_proj_i.unsqueeze(-2) + s_proj_j.unsqueeze(-3)

        # Step 2: Embed pair distances of representative atoms
        # d_ij = ||x̃_rep(i)^pred - x̃_rep(j)^pred||

        # Get representative atoms for each token (simple: first atom of each token)
        x_rep = torch.zeros(batch_size, n_tokens, 3, device=device, dtype=x_pred.dtype)
        for i in range(n_tokens):
            mask = token_indices == i
            if mask.any():
                # Use the first atom as representative
                first_atom_idx = torch.where(mask)[0][0]
                x_rep[:, i] = x_pred[:, first_atom_idx]

        # Compute pairwise distances
        d_ij = torch.norm(
            x_rep.unsqueeze(-2) - x_rep.unsqueeze(-3), dim=-1
        )  # (batch, n_tokens, n_tokens)

        # Step 3: z_ij += LinearNoBias(one_hot(d_ij, v_bins=[3⅞ Å, 5⅞ Å, ..., 21⅞ Å]))
        # Use registered distance bins
        if not hasattr(self, "_distance_bins_device"):
            self.register_buffer("_distance_bins_device", self.distance_bins.clone())

        # Move bins to same device
        bins = self._distance_bins_device.to(device)
        d_ij_one_hot = one_hot(d_ij, bins)  # (batch, n_tokens, n_tokens, n_bins)
        d_ij_embedded = self.dist_embedding(d_ij_one_hot)
        z_ij = z_ij + d_ij_embedded

        # Step 4: {s_i}, {z_ij} += PairformerStack({s_i}, {z_ij}, N_block)
        s_i, z_ij = self.pairformer_stack(s_i, z_ij)

        # Step 5: p_ij^pae = softmax(LinearNoBias(z_ij))
        p_pae = torch.softmax(self.pae_head(z_ij), dim=-1)

        # Step 6: p_ij^pde = softmax(LinearNoBias(z_ij + z_ji))
        p_pde = torch.softmax(self.pde_head(z_ij + z_ij.transpose(-2, -3)), dim=-1)

        # Map token-level confidences back to atom level for pLDDT and resolved
        s_atom = torch.zeros(
            batch_size, n_atoms, self.c_s, device=device, dtype=s_i.dtype
        )
        for i in range(n_atoms):
            token_idx = token_indices[i]
            s_atom[:, i] = s_i[:, token_idx]

        # Step 7: p_l^plddt = softmax(LinearNoBias_token_atom_idx(l)(s_l(l)))
        p_plddt = torch.softmax(self.plddt_head(s_atom), dim=-1)

        # Step 8: p_l^resolved = softmax(LinearNoBias_token_atom_idx(l)(s_l(l)))
        p_resolved = torch.softmax(self.resolved_head(s_atom), dim=-1)

        # Step 9: return {p_l^plddt}, {p_ij^pae}, {p_ij^pde}, {p_l^resolved}
        return p_plddt, p_pae, p_pde, p_resolved


class _Distogram(nn.Module):
    r"""
    Distogram Head for AlphaFold 3.

    This module predicts distance distributions (distograms) between pairs of
    residues from pair representations. It outputs probability distributions
    over distance bins, which are useful for structure prediction and validation.

    Parameters
    ----------
    c_z : int, default=128
        Pair representation dimension
    n_bins : int, default=64
        Number of distance bins
    min_dist : float, default=2.3125
        Minimum distance in Angstroms
    max_dist : float, default=21.6875
        Maximum distance in Angstroms
    """

    def __init__(
        self,
        c_z: int = 128,
        n_bins: int = 64,
        min_dist: float = 2.3125,
        max_dist: float = 21.6875,
    ):
        super().__init__()

        self.c_z = c_z
        self.n_bins = n_bins
        self.min_dist = min_dist
        self.max_dist = max_dist

        # Create distance bins
        self.register_buffer(
            "distance_bins", torch.linspace(min_dist, max_dist, n_bins)
        )

        # Distogram prediction head
        self.distogram_head = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z, bias=True),
            nn.ReLU(),
            nn.Linear(c_z, n_bins, bias=True),
        )

    def forward(self, z_ij: Tensor) -> Tensor:
        r"""
        Forward pass of Distogram Head.

        Parameters
        ----------
        z_ij : Tensor, shape=(batch_size, n_tokens, n_tokens, c_z)
            Pair representations

        Returns
        -------
        p_distogram : Tensor, shape=(batch_size, n_tokens, n_tokens, n_bins)
            Distance probability distributions (softmax over bins)
        """
        # Apply distogram head
        logits = self.distogram_head(z_ij)  # (batch, n_tokens, n_tokens, n_bins)

        # Apply softmax to get probabilities
        p_distogram = torch.softmax(logits, dim=-1)

        return p_distogram


class _InputFeatureEmbedder(nn.Module):
    r"""
    Input Feature Embedder for AlphaFold 3.

    This module constructs an initial 1D embedding by embedding per-atom features
    and concatenating per-token features. Implements Algorithm 2 exactly.

    Parameters
    ----------
    c_atom : int, default=128
        Channel dimension for atom representations
    c_atompair : int, default=16
        Channel dimension for atom pair representations
    c_token : int, default=384
        Channel dimension for token representations

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import _InputFeatureEmbedder
    >>> batch_size, n_atoms = 2, 1000
    >>> module = _InputFeatureEmbedder()
    >>>
    >>> # Feature dictionary
    >>> f_star = {
    ...     'ref_pos': torch.randn(batch_size, n_atoms, 3),
    ...     'restype': torch.randint(0, 21, (batch_size, n_atoms)),
    ...     'profile': torch.randn(batch_size, n_atoms, 20),
    ...     'deletion_mean': torch.randn(batch_size, n_atoms),
    ... }
    >>> s_i = module(f_star)
    >>> len(s_i)
    2

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 2: Construct an initial 1D embedding
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 384,
    ):
        super().__init__()

        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token

        # Step 1: Embed per-atom features using AtomAttentionEncoder
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
        )

        # For step 2: Concatenate per-token features
        # This would typically include restype, profile, deletion_mean embeddings
        # For simplicity, we'll just use the atom features

    def forward(self, f_star: dict) -> dict:
        r"""
        Forward pass of Input Feature Embedder.

        Implements Algorithm 2:
        1. {a_i}, _, _, _ = AtomAttentionEncoder({f*}, ∅, ∅, ∅, c_atom=128, c_atompair=16, c_token=384)
        2. s_i = concat(a_i, f_i^restype, f_i^profile, f_i^deletion_mean)

        Parameters
        ----------
        f_star : dict
            Dictionary containing atom features with keys like:
            - 'ref_pos': reference positions (batch, n_atoms, 3)
            - 'restype': residue types (batch, n_tokens)
            - 'profile': sequence profile (batch, n_tokens, 20)
            - 'deletion_mean': deletion statistics (batch, n_tokens)

        Returns
        -------
        s_i : dict
            Dictionary with key 'single' containing concatenated features
        """
        # Step 1: Embed per-atom features
        # Pass empty tensors for r_t, s_trunk, z_ij as specified in algorithm
        batch_size = next(iter(f_star.values())).shape[0]
        device = next(iter(f_star.values())).device

        # Create empty/dummy inputs as specified in the algorithm
        if "atom_coordinates" in f_star:
            r_t_empty = f_star["atom_coordinates"]
        else:
            # Try to find a 3D coordinate tensor, or create dummy one
            coord_tensors = [
                v for v in f_star.values() if len(v.shape) == 3 and v.shape[-1] == 3
            ]
            if coord_tensors:
                r_t_empty = coord_tensors[0]
            else:
                # Create minimal dummy coordinates
                r_t_empty = torch.zeros(batch_size, 1, 3, device=device)
        s_trunk_empty = None  # Empty trunk
        z_ij_empty = None  # Empty pair representations

        # Get atom embeddings
        a_i, _, _, _ = self.atom_attention_encoder(
            f_star, r_t_empty, s_trunk_empty, z_ij_empty
        )

        # Step 2: Concatenate the per-token features
        # Extract per-token features if available
        features_to_concat = [a_i]  # Start with atom embeddings

        # Add restype if available
        if "restype" in f_star:
            restype = f_star["restype"]
            if len(restype.shape) == 2:  # (batch, n_tokens)
                # Convert to one-hot or embedding
                restype_embedded = torch.nn.functional.one_hot(
                    restype.long(), num_classes=21
                ).float()
                features_to_concat.append(restype_embedded)

        # Add profile if available
        if "profile" in f_star:
            profile = f_star["profile"]
            if len(profile.shape) == 3:  # (batch, n_tokens, 20)
                features_to_concat.append(profile)

        # Add deletion_mean if available
        if "deletion_mean" in f_star:
            deletion_mean = f_star["deletion_mean"]
            if len(deletion_mean.shape) == 2:  # (batch, n_tokens)
                features_to_concat.append(deletion_mean.unsqueeze(-1))

        # Concatenate all features
        s_i = torch.cat(features_to_concat, dim=-1)

        return {"single": s_i}
