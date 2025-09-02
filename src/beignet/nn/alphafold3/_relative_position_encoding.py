import torch
import torch.nn as nn
from torch import Tensor

from ..._one_hot import one_hot


class RelativePositionEncoding(nn.Module):
    r"""
    Relative Position Encoding for AlphaFold 3.

    This module implements Algorithm 3 exactly, computing relative position
    encodings based on asymmetric ID, residue index, entity ID, token index,
    and chain ID information.

    Parameters
    ----------
    r_max : int, default=32
        Maximum residue separation for clipping
    s_max : int, default=2
        Maximum chain separation for clipping
    c_z : int, default=128
        Output channel dimension

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import RelativePositionEncoding
    >>> batch_size, n_tokens = 2, 100
    >>> module = RelativePositionEncoding()
    >>> f_star = {
    ...     'asym_id': torch.randint(0, 5, (batch_size, n_tokens)),
    ...     'residue_index': torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
    ...     'entity_id': torch.randint(0, 3, (batch_size, n_tokens)),
    ...     'token_index': torch.arange(n_tokens).unsqueeze(0).expand(batch_size, -1),
    ...     'sym_id': torch.randint(0, 10, (batch_size, n_tokens)),
    ... }
    >>> p_ij = module(f_star)
    >>> p_ij.shape
    torch.Size([2, 100, 100, 128])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 3: Relative position encoding
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128):
        super().__init__()

        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z

        # Final linear projection
        # 2 rel distance features (residue, token) * (2*r_max+2) + 1 same_entity + 1 chain feature * (2*s_max+2)
        self.linear = nn.Linear(
            2 * (2 * r_max + 2) + 1 + (2 * s_max + 2), c_z, bias=False
        )

    def forward(self, f_star: dict) -> Tensor:
        r"""
        Forward pass implementing Algorithm 3 exactly.

        Parameters
        ----------
        f_star : dict
            Dictionary containing features with keys:
            - 'asym_id': asymmetric unit IDs (batch, n_tokens)
            - 'residue_index': residue indices (batch, n_tokens)
            - 'entity_id': entity IDs (batch, n_tokens)
            - 'token_index': token indices (batch, n_tokens)
            - 'sym_id': symmetry IDs (batch, n_tokens)

        Returns
        -------
        p_ij : Tensor, shape=(batch, n_tokens, n_tokens, c_z)
            Relative position encodings
        """
        # Extract features
        asym_id_i = f_star["asym_id"]  # (batch, n_tokens)
        residue_index_i = f_star["residue_index"]
        entity_id_i = f_star["entity_id"]
        token_index_i = f_star["token_index"]
        sym_id_i = f_star["sym_id"]

        batch_size, n_tokens = asym_id_i.shape
        device = asym_id_i.device

        # Create pairwise comparisons
        asym_id_j = asym_id_i.unsqueeze(-1)  # (batch, n_tokens, 1)
        asym_id_i = asym_id_i.unsqueeze(-2)  # (batch, 1, n_tokens)

        residue_index_j = residue_index_i.unsqueeze(-1)
        residue_index_i = residue_index_i.unsqueeze(-2)

        entity_id_j = entity_id_i.unsqueeze(-1)
        entity_id_i = entity_id_i.unsqueeze(-2)

        token_index_j = token_index_i.unsqueeze(-1)
        token_index_i = token_index_i.unsqueeze(-2)

        sym_id_j = sym_id_i.unsqueeze(-1)
        sym_id_i = sym_id_i.unsqueeze(-2)

        # Step 1: b_ij^same_chain = (f_i^asym_id == f_j^asym_id)
        b_same_chain = (asym_id_i == asym_id_j).float()

        # Step 2: b_ij^same_residue = (f_i^residue_index == f_j^residue_index)
        b_same_residue = (residue_index_i == residue_index_j).float()

        # Step 3: b_ij^same_entity = (f_i^entity_id == f_j^entity_id)
        b_same_entity = (entity_id_i == entity_id_j).float()

        # Step 4: Relative residue distance
        d_residue = torch.where(
            b_same_chain.bool(),
            torch.clamp(
                residue_index_i - residue_index_j + self.r_max, 0, 2 * self.r_max
            ),
            2 * self.r_max + 1,
        ).long()

        # Step 5: One-hot encode residue distance
        a_rel_pos = one_hot(
            d_residue.float(),
            torch.arange(2 * self.r_max + 2, device=device, dtype=torch.float32),
        )  # (batch, n_tokens, n_tokens, 2*r_max+2)

        # Step 6: Relative token distance
        d_token = torch.where(
            (b_same_chain.bool()) & (b_same_residue.bool()),
            torch.clamp(token_index_i - token_index_j + self.r_max, 0, 2 * self.r_max),
            2 * self.r_max + 1,
        ).long()

        # Step 7: One-hot encode token distance
        a_rel_token = one_hot(
            d_token.float(),
            torch.arange(2 * self.r_max + 2, device=device, dtype=torch.float32),
        )  # (batch, n_tokens, n_tokens, 2*r_max+2)

        # Step 8: Relative chain distance
        d_chain = torch.where(
            ~b_same_chain.bool(),
            torch.clamp(sym_id_i - sym_id_j + self.s_max, 0, 2 * self.s_max),
            2 * self.s_max + 1,
        ).long()

        # Step 9: One-hot encode chain distance
        a_rel_chain = one_hot(
            d_chain.float(),
            torch.arange(2 * self.s_max + 2, device=device, dtype=torch.float32),
        )  # (batch, n_tokens, n_tokens, 2*s_max+2)

        # Step 10: Concatenate all features and apply linear projection
        all_features = torch.cat(
            [
                a_rel_pos,  # (batch, n_tokens, n_tokens, 2*r_max+2)
                a_rel_token,  # (batch, n_tokens, n_tokens, 2*r_max+2)
                b_same_entity.unsqueeze(-1),  # (batch, n_tokens, n_tokens, 1)
                a_rel_chain,  # (batch, n_tokens, n_tokens, 2*s_max+2)
            ],
            dim=-1,
        )

        # Step 11: Linear projection
        p_ij = self.linear(all_features)

        return p_ij
