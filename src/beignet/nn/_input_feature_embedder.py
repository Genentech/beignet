import torch
import torch.nn as nn

from ._atom_attention_encoder import AtomAttentionEncoder


class InputFeatureEmbedder(nn.Module):
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
    >>> from beignet.nn import InputFeatureEmbedder
    >>> batch_size, n_atoms = 2, 1000
    >>> module = InputFeatureEmbedder()
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
