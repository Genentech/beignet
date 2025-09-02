import torch.nn as nn
from torch import Tensor

from ._attention_pair_bias import AttentionPairBias
from ._transition import Transition
from ._triangle_attention_ending_node import TriangleAttentionEndingNode
from ._triangle_attention_starting_node import TriangleAttentionStartingNode
from ._triangle_multiplication_incoming import TriangleMultiplicationIncoming
from ._triangle_multiplication_outgoing import TriangleMultiplicationOutgoing


class PairformerStackBlock(nn.Module):
    r"""
    Single block of the Pairformer stack from AlphaFold 3 Algorithm 17.

    This implements exactly one iteration of the loop in Algorithm 17,
    following the precise order and dropout specifications.

    Parameters
    ----------
    c_s : int, default=384
        Channel dimension for single representation
    c_z : int, default=128
        Channel dimension for pair representation
    n_head_single : int, default=16
        Number of attention heads for single representation (as per algorithm)
    n_head_pair : int, default=4
        Number of attention heads for pair representation
    dropout_rate : float, default=0.25
        Dropout rate (as specified in Algorithm 17)
    transition_n : int, default=4
        Multiplier for transition layer hidden dimension

    Examples
    --------
    >>> import torch
    >>> from beignet.nn._pairformer_stack import PairformerStackBlock
    >>> batch_size, seq_len = 2, 10
    >>> c_s, c_z = 384, 128
    >>> module = PairformerStackBlock(c_s=c_s, c_z=c_z)
    >>> s_i = torch.randn(batch_size, seq_len, c_s)
    >>> z_ij = torch.randn(batch_size, seq_len, seq_len, c_z)
    >>> s_out, z_out = module(s_i, z_ij)
    >>> s_out.shape
    torch.Size([2, 10, 384])
    >>> z_out.shape
    torch.Size([2, 10, 10, 128])
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        n_head_single: int = 16,
        n_head_pair: int = 4,
        dropout_rate: float = 0.25,
        transition_n: int = 4,
    ):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.dropout_rate = dropout_rate

        # Pair representation processing (steps 2-6)
        self.triangle_mult_outgoing = TriangleMultiplicationOutgoing(c=c_z)
        self.triangle_mult_incoming = TriangleMultiplicationIncoming(c=c_z)
        self.triangle_attention_starting = TriangleAttentionStartingNode(
            c=c_z, n_head=n_head_pair
        )
        self.triangle_attention_ending = TriangleAttentionEndingNode(
            c=c_z, n_head=n_head_pair
        )
        self.pair_transition = Transition(c=c_z, n=transition_n)

        # Single representation processing (steps 7-8)
        self.attention_pair_bias = AttentionPairBias(
            c_a=c_s, c_s=c_s, c_z=c_z, n_head=n_head_single
        )
        self.single_transition = Transition(c=c_s, n=transition_n)

        # Dropout layers
        self.dropout_rowwise = nn.Dropout(dropout_rate)
        self.dropout_columnwise = nn.Dropout(dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, s_i: Tensor, z_ij: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Forward pass implementing Algorithm 17 exactly.

        Parameters
        ----------
        s_i : Tensor, shape=(..., s, c_s)
            Single representation
        z_ij : Tensor, shape=(..., s, s, c_z)
            Pair representation

        Returns
        -------
        s_out : Tensor, shape=(..., s, c_s)
            Updated single representation
        z_out : Tensor, shape=(..., s, s, c_z)
            Updated pair representation
        """
        # Algorithm 17, steps 2-6: Pair stack processing

        # Step 2: Triangle multiplication outgoing with row-wise dropout
        z_ij = z_ij + self.dropout_rowwise(self.triangle_mult_outgoing(z_ij))

        # Step 3: Triangle multiplication incoming with row-wise dropout
        z_ij = z_ij + self.dropout_rowwise(self.triangle_mult_incoming(z_ij))

        # Step 4: Triangle attention starting node with row-wise dropout
        z_ij = z_ij + self.dropout_rowwise(self.triangle_attention_starting(z_ij))

        # Step 5: Triangle attention ending node with column-wise dropout
        z_ij = z_ij + self.dropout_columnwise(self.triangle_attention_ending(z_ij))

        # Step 6: Pair transition
        z_ij = z_ij + self.pair_transition(z_ij)

        # Steps 7-8: Single representation processing

        # Step 7: Attention with pair bias (θij = 0, Nhead = 16)
        s_i = s_i + self.attention_pair_bias(s_i, z_ij)

        # Step 8: Single transition
        s_i = s_i + self.single_transition(s_i)

        return s_i, z_ij


class PairformerStack(nn.Module):
    r"""
    Pairformer stack from AlphaFold 3 Algorithm 17.

    This is the exact implementation of the Pairformer stack as specified
    in Algorithm 17, which processes single and pair representations through
    N_block iterations of triangle operations and attention mechanisms.

    Parameters
    ----------
    n_block : int, default=48
        Number of Pairformer blocks (N_block in Algorithm 17)
    c_s : int, default=384
        Channel dimension for single representation
    c_z : int, default=128
        Channel dimension for pair representation
    n_head_single : int, default=16
        Number of attention heads for single representation
    n_head_pair : int, default=4
        Number of attention heads for pair representation
    dropout_rate : float, default=0.25
        Dropout rate as specified in Algorithm 17
    transition_n : int, default=4
        Multiplier for transition layer hidden dimension

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import PairformerStack
    >>> batch_size, seq_len = 2, 10
    >>> n_block, c_s, c_z = 4, 384, 128
    >>> module = PairformerStack(n_block=n_block, c_s=c_s, c_z=c_z)
    >>> s_i = torch.randn(batch_size, seq_len, c_s)
    >>> z_ij = torch.randn(batch_size, seq_len, seq_len, c_z)
    >>> s_out, z_out = module(s_i, z_ij)
    >>> s_out.shape
    torch.Size([2, 10, 384])
    >>> z_out.shape
    torch.Size([2, 10, 10, 128])

    References
    ----------
    .. [1] Abramson, J., Adler, J., Dunger, J. et al. Accurate structure prediction
           of biomolecular interactions with AlphaFold 3. Nature 630, 493–500 (2024).
           Algorithm 17: Pairformer stack
    """

    def __init__(
        self,
        n_block: int = 48,
        c_s: int = 384,
        c_z: int = 128,
        n_head_single: int = 16,
        n_head_pair: int = 4,
        dropout_rate: float = 0.25,
        transition_n: int = 4,
    ):
        super().__init__()

        self.n_block = n_block
        self.c_s = c_s
        self.c_z = c_z

        # Create n_block Pairformer stack blocks (each with its own parameters)
        self.blocks = nn.ModuleList(
            [
                PairformerStackBlock(
                    c_s=c_s,
                    c_z=c_z,
                    n_head_single=n_head_single,
                    n_head_pair=n_head_pair,
                    dropout_rate=dropout_rate,
                    transition_n=transition_n,
                )
                for _ in range(n_block)
            ]
        )

    def forward(self, s_i: Tensor, z_ij: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Forward pass of Pairformer stack.

        Parameters
        ----------
        s_i : Tensor, shape=(..., s, c_s)
            Single representation where s is sequence length
        z_ij : Tensor, shape=(..., s, s, c_z)
            Pair representation

        Returns
        -------
        s_out : Tensor, shape=(..., s, c_s)
            Updated single representation after all blocks
        z_out : Tensor, shape=(..., s, s, c_z)
            Updated pair representation after all blocks
        """
        # Validate input shapes
        if s_i.shape[-2] != z_ij.shape[-2] or s_i.shape[-2] != z_ij.shape[-3]:
            raise ValueError(
                f"Sequence length mismatch: single representation has {s_i.shape[-2]} "
                f"residues but pair representation has shape {z_ij.shape[-3:]}"
            )

        if s_i.shape[-1] != self.c_s:
            raise ValueError(
                f"Single representation has {s_i.shape[-1]} channels, "
                f"expected {self.c_s}"
            )

        if z_ij.shape[-1] != self.c_z:
            raise ValueError(
                f"Pair representation has {z_ij.shape[-1]} channels, "
                f"expected {self.c_z}"
            )

        # Algorithm 17: for all l ∈ [1, ..., N_block] do
        for block in self.blocks:
            s_i, z_ij = block(s_i, z_ij)

        # Algorithm 17 step 10: return {s_i}, {z_ij}
        return s_i, z_ij
