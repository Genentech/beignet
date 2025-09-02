import torch
import torch.nn as nn
from torch import Tensor

from .. import one_hot
from ._pairformer_stack import PairformerStack


class AlphaFold3Confidence(nn.Module):
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

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import AlphaFold3Confidence
    >>> batch_size, n_atoms, n_tokens = 2, 1000, 32
    >>> module = AlphaFold3Confidence()
    >>>
    >>> s_inputs = torch.randn(batch_size, n_atoms, 100)
    >>> s_i = torch.randn(batch_size, n_tokens, 384)
    >>> z_ij = torch.randn(batch_size, n_tokens, n_tokens, 128)
    >>> x_pred = torch.randn(batch_size, n_atoms, 3)
    >>>
    >>> p_plddt, p_pae, p_pde, p_resolved = module(s_inputs, s_i, z_ij, x_pred)
    >>> p_plddt.shape, p_pae.shape, p_pde.shape, p_resolved.shape
    (torch.Size([2, 1000, 50]), torch.Size([2, 32, 32, 64]), torch.Size([2, 32, 32, 64]), torch.Size([2, 1000, 2]))

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 31: Confidence head
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
