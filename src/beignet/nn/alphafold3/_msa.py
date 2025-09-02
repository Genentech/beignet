import torch
import torch.nn as nn
from torch import Tensor

from ._transition import Transition
from ._triangle_attention_ending_node import TriangleAttentionEndingNode
from ._triangle_attention_starting_node import TriangleAttentionStartingNode
from ._triangle_multiplication_incoming import TriangleMultiplicationIncoming
from ._triangle_multiplication_outgoing import TriangleMultiplicationOutgoing


class MSA(nn.Module):
    r"""
    Multiple Sequence Alignment Module from AlphaFold 3.

    This implements Algorithm 8 from AlphaFold 3, which is a complete MSA processing
    module that combines multiple sub-modules in a structured way:

    1. MSA representation initialization and random sampling
    2. Communication block with OuterProductMean
    3. MSA stack with MSAPairWeightedAveraging and Transition
    4. Pair stack with triangle updates and attention

    The module processes MSA features, single representations, and pair representations
    through multiple blocks to capture complex evolutionary and structural patterns.

    Parameters
    ----------
    n_block : int, default=4
        Number of processing blocks
    c_m : int, default=64
        Channel dimension for MSA representation
    c_z : int, default=128
        Channel dimension for pair representation
    c_s : int, default=256
        Channel dimension for single representation
    n_head_msa : int, default=8
        Number of attention heads for MSA operations
    n_head_pair : int, default=4
        Number of attention heads for pair operations
    dropout_rate : float, default=0.15
        Dropout rate for MSA operations

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import MSA
    >>> batch_size, seq_len, n_seq = 2, 32, 16
    >>> c_m, c_z, c_s = 64, 128, 256
    >>>
    >>> module = MSA(n_block=2, c_m=c_m, c_z=c_z, c_s=c_s)
    >>>
    >>> # Input features
    >>> f_msa = torch.randn(batch_size, seq_len, n_seq, 23)  # MSA features
    >>> f_has_deletion = torch.randn(batch_size, seq_len, n_seq, 1)
    >>> f_deletion_value = torch.randn(batch_size, seq_len, n_seq, 1)
    >>> s_inputs = torch.randn(batch_size, seq_len, c_s)  # Single inputs
    >>> z_ij = torch.randn(batch_size, seq_len, seq_len, c_z)  # Pair representation
    >>>
    >>> # Forward pass
    >>> updated_z_ij = module(f_msa, f_has_deletion, f_deletion_value, s_inputs, z_ij)
    >>> updated_z_ij.shape
    torch.Size([2, 32, 32, 128])

    References
    ----------
    .. [1] AlphaFold 3 paper, Algorithm 8: MSA Module
    """

    def __init__(
        self,
        n_block: int = 4,
        c_m: int = 64,
        c_z: int = 128,
        c_s: int = 256,
        n_head_msa: int = 8,
        n_head_pair: int = 4,
        dropout_rate: float = 0.15,
    ):
        super().__init__()

        self.n_block = n_block
        self.c_m = c_m
        self.c_z = c_z
        self.c_s = c_s
        self.n_head_msa = n_head_msa
        self.n_head_pair = n_head_pair
        self.dropout_rate = dropout_rate

        # Step 3: Initial linear projection for MSA (concatenated features -> c_m)
        # Input features: f_msa (23) + f_has_deletion (1) + f_deletion_value (1) = 25 channels
        self.msa_linear = nn.Linear(25, c_m, bias=False)

        # Step 4: Linear projection for single inputs (s_inputs -> c_m)
        self.single_linear = nn.Linear(c_s, c_m, bias=False)

        # Communication: OuterProductMean (step 6)
        self.outer_product_mean = _OuterProductMean(c=c_m, c_z=c_z)

        # MSA stack components (step 7-8)
        self.msa_pair_weighted_averaging = _MSAPairWeightedAveraging(
            c_m=c_m, c_z=c_z, n_head=n_head_msa
        )
        self.msa_transition = Transition(c=c_m, n=4)
        self.msa_dropout = nn.Dropout(dropout_rate)

        # Pair stack components (step 9-13)
        self.triangle_mult_outgoing = TriangleMultiplicationOutgoing(c=c_z)
        self.triangle_mult_incoming = TriangleMultiplicationIncoming(c=c_z)
        self.triangle_attention_starting = TriangleAttentionStartingNode(
            c=c_z, n_head=n_head_pair
        )
        self.triangle_attention_ending = TriangleAttentionEndingNode(
            c=c_z, n_head=n_head_pair
        )
        self.pair_transition = Transition(c=c_z, n=4)

        # Dropout layers for pair operations
        self.pair_dropout_rowwise = nn.Dropout(0.25)  # For steps 9,10,11
        self.pair_dropout_columnwise = nn.Dropout(0.25)  # For step 12

    def forward(
        self,
        f_msa: Tensor,
        f_has_deletion: Tensor,
        f_deletion_value: Tensor,
        s_inputs: Tensor,
        z_ij: Tensor,
    ) -> Tensor:
        r"""
        Forward pass of the MSA Module.

        Parameters
        ----------
        f_msa : Tensor, shape=(..., s, n_seq, 23)
            MSA features (amino acid profiles)
        f_has_deletion : Tensor, shape=(..., s, n_seq, 1)
            Has deletion features
        f_deletion_value : Tensor, shape=(..., s, n_seq, 1)
            Deletion value features
        s_inputs : Tensor, shape=(..., s, c_s)
            Single representation inputs
        z_ij : Tensor, shape=(..., s, s, c_z)
            Pair representation

        Returns
        -------
        z_ij : Tensor, shape=(..., s, s, c_z)
            Updated pair representation
        """
        # Input validation
        seq_len_msa = f_msa.shape[-3]
        seq_len_single = s_inputs.shape[-2]
        n_seq_msa = f_msa.shape[-2]

        if seq_len_msa != seq_len_single:
            raise ValueError(
                f"Sequence length mismatch: MSA has {seq_len_msa} residues, "
                f"single representation has {seq_len_single} residues"
            )

        # Check MSA feature consistency
        if (
            f_has_deletion.shape[-3] != seq_len_msa
            or f_deletion_value.shape[-3] != seq_len_msa
        ):
            raise ValueError("All MSA features must have the same sequence length")

        if (
            f_has_deletion.shape[-2] != n_seq_msa
            or f_deletion_value.shape[-2] != n_seq_msa
        ):
            raise ValueError("All MSA features must have the same number of sequences")

        # Check pair representation compatibility
        if z_ij.shape[-2] != seq_len_msa or z_ij.shape[-3] != seq_len_msa:
            raise ValueError(
                f"Pair representation shape mismatch: expected ({seq_len_msa}, {seq_len_msa}), "
                f"got ({z_ij.shape[-3]}, {z_ij.shape[-2]})"
            )

        m_si = self.msa_linear(
            torch.concatenate(
                [
                    f_msa,
                    f_has_deletion,
                    f_deletion_value,
                ],
                dim=-1,
            )
        ) + torch.unsqueeze(self.single_linear(s_inputs), -2)

        # Step 5: Process through N_block iterations
        for _ in range(self.n_block):
            # Step 6: Communication - OuterProductMean
            # OuterProductMean now properly handles MSA sequences and computes mean over outer products
            # Pass the full MSA representation to capture coevolutionary information
            z_ij = z_ij + self.outer_product_mean(m_si)  # m_si: (..., s, n_seq, c_m)

            # MSA stack (steps 7-8)
            # Step 7: MSA Pair Weighted Averaging with dropout
            m_si = m_si + self.msa_dropout(self.msa_pair_weighted_averaging(m_si, z_ij))

            # Step 8: MSA Transition
            m_si = m_si + self.msa_transition(m_si)

            # Pair stack (steps 9-13)
            # Step 9: Triangle Multiplication Outgoing with rowwise dropout
            z_ij = z_ij + self.pair_dropout_rowwise(self.triangle_mult_outgoing(z_ij))

            # Step 10: Triangle Multiplication Incoming with rowwise dropout
            z_ij = z_ij + self.pair_dropout_rowwise(self.triangle_mult_incoming(z_ij))

            # Step 11: Triangle Attention Starting Node with rowwise dropout
            z_ij = z_ij + self.pair_dropout_rowwise(
                self.triangle_attention_starting(z_ij)
            )

            # Step 12: Triangle Attention Ending Node with columnwise dropout
            z_ij = z_ij + self.pair_dropout_columnwise(
                self.triangle_attention_ending(z_ij)
            )

            # Step 13: Pair Transition
            z_ij = z_ij + self.pair_transition(z_ij)

        return z_ij


class _OuterProductMean(nn.Module):
    r"""
    Outer Product Mean module from AlphaFold 3.

    This implements Algorithm 9 from AlphaFold 3, which computes outer products
    between MSA sequence representations and applies linear transformations.
    The "mean" in outer product mean refers to averaging outer products computed
    across all MSA sequences, capturing coevolutionary information.

    The algorithm follows these steps:
    1. Apply layer normalization to input MSA representation
    2. Compute linear projections without bias to get two vectors for each sequence
    3. For each MSA sequence s, compute pairwise outer products a_s,i ⊗ b_s,j
       for all residue pairs (i, j), flatten the (c × c) matrix to length c*c
    4. Average outer products over all MSA sequences (this is the "mean")
    5. Apply final linear transformation to produce pair features

    Parameters
    ----------
    c : int, default=32
        Input channel dimension
    c_z : int, default=128
        Output channel dimension for final projection
    """

    def __init__(self, c: int = 32, c_z: int = 128):
        super().__init__()

        self.c = c
        self.c_z = c_z

        # Layer normalization (step 1)
        self.layer_norm = nn.LayerNorm(c)

        # Linear projection without bias to get a and b vectors (step 2)
        self.linear_no_bias = nn.Linear(c, 2 * c, bias=False)

        # Final linear projection (step 4)
        self.final_linear = nn.Linear(c * c, c_z)

    def forward(self, m_si: Tensor) -> Tensor:
        r"""
        Forward pass of outer product mean.

        Parameters
        ----------
        m_si : Tensor, shape=(..., s, n_seq, c)
            Input MSA representation where s is sequence length,
            n_seq is number of MSA sequences, and c is channel dimension.

        Returns
        -------
        z_ij : Tensor, shape=(..., s, s, c_z)
            Pairwise representation after outer product mean operation.
        """
        m_si = self.layer_norm(m_si)  # (..., s, n_seq, c)

        a_si, b_si = torch.chunk(
            self.linear_no_bias(m_si),
            2,
            dim=-1,
        )

        return self.final_linear(
            torch.mean(
                torch.flatten(
                    torch.unsqueeze(torch.unsqueeze(a_si, -3), -1)
                    * torch.unsqueeze(torch.unsqueeze(b_si, -4), -2),
                    start_dim=-2,
                ),
                dim=-2,
            )
        )


class _MSAPairWeightedAveraging(nn.Module):
    r"""
    MSA pair weighted averaging with gating from AlphaFold 3.

    This implements Algorithm 10 from AlphaFold 3, which performs weighted
    averaging of MSA representations using pair representations as weights,
    with gating for controlled information flow.

    Parameters
    ----------
    c_m : int, default=32
        Channel dimension for the MSA representation
    c_z : int, default=128
        Channel dimension for the pair representation
    n_head : int, default=8
        Number of attention heads

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import _MSAPairWeightedAveraging
    >>> batch_size, seq_len, n_seq, c_m, c_z = 2, 10, 5, 32, 128
    >>> n_head = 8
    >>> module = _MSAPairWeightedAveraging(c_m=c_m, c_z=c_z, n_head=n_head)
    >>> m_si = torch.randn(batch_size, seq_len, n_seq, c_m)
    >>> z_ij = torch.randn(batch_size, seq_len, seq_len, c_z)
    >>> m_tilde_si = module(m_si, z_ij)
    >>> m_tilde_si.shape
    torch.Size([2, 10, 5, 32])

    References
    ----------
    .. [1] AlphaFold 3 paper, Algorithm 10: MSA pair weighted averaging with gating
    """

    def __init__(self, c_m: int = 32, c_z: int = 128, n_head: int = 8):
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.n_head = n_head
        self.head_dim = c_m // n_head

        if c_m % n_head != 0:
            raise ValueError(
                f"MSA channel dimension {c_m} must be divisible by number of heads {n_head}"
            )

        # Layer normalization for MSA input (step 1)
        self.msa_layer_norm = nn.LayerNorm(c_m)

        # Linear projection for MSA values (step 2)
        self.linear_v = nn.Linear(c_m, c_m, bias=False)

        # Linear projection for pair weights (step 3)
        self.linear_b = nn.Linear(c_z, n_head, bias=False)
        self.pair_layer_norm = nn.LayerNorm(c_z)

        # Gate projection for MSA (step 4)
        self.linear_g = nn.Linear(c_m, c_m, bias=False)

        # Output projection (step 7)
        self.output_linear = nn.Linear(c_m, c_m, bias=False)

    def forward(self, m_si: Tensor, z_ij: Tensor) -> Tensor:
        r"""
        Forward pass of MSA pair weighted averaging with gating.

        Parameters
        ----------
        m_si : Tensor, shape=(..., s, n_seq, c_m)
            Input MSA representation where s is sequence length,
            n_seq is number of sequences, and c_m is MSA channel dimension.
        z_ij : Tensor, shape=(..., s, s, c_z)
            Input pair representation where s is sequence length
            and c_z is pair channel dimension.

        Returns
        -------
        m_tilde_si : Tensor, shape=(..., s, n_seq, c_m)
            Updated MSA representation after pair-weighted averaging.
        """
        m_si = self.msa_layer_norm(m_si)

        return self.output_linear(
            (
                torch.sigmoid(self.linear_g(m_si)).view(
                    *(m_si.shape[:-3]),
                    m_si.shape[-3],
                    m_si.shape[-2],
                    self.n_head,
                    self.head_dim,
                )
                * torch.einsum(
                    "...ijh,...jnhd->...inhd",
                    torch.softmax(
                        self.linear_b(self.pair_layer_norm(z_ij)),
                        dim=-2,
                    ),
                    self.linear_v(m_si).view(
                        *(m_si.shape[:-3]),
                        m_si.shape[-3],
                        m_si.shape[-2],
                        self.n_head,
                        self.head_dim,
                    ),
                )
            ).view(
                *(m_si.shape[:-3]),
                m_si.shape[-3],
                m_si.shape[-2],
                self.c_m,
            )
        )
