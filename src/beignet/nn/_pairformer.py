import torch.nn as nn
from torch import Tensor

from ._transition import Transition
from ._triangle_attention_ending_node import TriangleAttentionEndingNode
from ._triangle_attention_starting_node import TriangleAttentionStartingNode
from ._triangle_multiplication_incoming import TriangleMultiplicationIncoming
from ._triangle_multiplication_outgoing import TriangleMultiplicationOutgoing


class SingleRowAttention(nn.Module):
    r"""
    Single row attention for processing single representation in Pairformer.

    This performs self-attention on the single representation, which has only
    one row (unlike MSA representations that have multiple sequence rows).

    Parameters
    ----------
    c_s : int, default=384
        Channel dimension for single representation
    n_head : int, default=8
        Number of attention heads
    dropout_rate : float, default=0.15
        Dropout rate for attention

    Examples
    --------
    >>> import torch
    >>> from beignet.nn._pairformer import SingleRowAttention
    >>> batch_size, seq_len, c_s = 2, 10, 384
    >>> n_head = 8
    >>> module = SingleRowAttention(c_s=c_s, n_head=n_head)
    >>> s_i = torch.randn(batch_size, seq_len, c_s)
    >>> s_out = module(s_i)
    >>> s_out.shape
    torch.Size([2, 10, 384])
    """

    def __init__(self, c_s: int = 384, n_head: int = 8, dropout_rate: float = 0.15):
        super().__init__()

        self.c_s = c_s
        self.n_head = n_head
        self.head_dim = c_s // n_head

        if c_s % n_head != 0:
            raise ValueError(
                f"Channel dimension {c_s} must be divisible by number of heads {n_head}"
            )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(c_s)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=c_s,
            num_heads=n_head,
            dropout=dropout_rate,
            batch_first=True,
        )

    def forward(self, s_i: Tensor) -> Tensor:
        r"""
        Forward pass of single row attention.

        Parameters
        ----------
        s_i : Tensor, shape=(..., s, c_s)
            Single representation where s is sequence length.

        Returns
        -------
        s_out : Tensor, shape=(..., s, c_s)
            Updated single representation.
        """
        batch_shape = s_i.shape[:-2]
        seq_len = s_i.shape[-2]

        # Flatten batch dimensions for attention
        s_flat = s_i.view(-1, seq_len, self.c_s)

        # Layer norm
        s_normed = self.layer_norm(s_flat)

        # Self-attention (query, key, value are all the same)
        attn_out, _ = self.attention(s_normed, s_normed, s_normed)

        # Reshape back to original batch shape
        s_out = attn_out.view(*batch_shape, seq_len, self.c_s)

        return s_out


class PairformerBlock(nn.Module):
    r"""
    Single block of the Pairformer module from AlphaFold 3.

    This implements one block that processes both single and pair representations
    through triangle operations, attention, and transitions, similar to
    Evoformer but simpler (no MSA processing, no outer product mean).

    Parameters
    ----------
    c_s : int, default=384
        Channel dimension for single representation
    c_z : int, default=128
        Channel dimension for pair representation
    n_head_single : int, default=8
        Number of attention heads for single representation
    n_head_pair : int, default=4
        Number of attention heads for pair representation
    dropout_rate : float, default=0.15
        Dropout rate
    transition_n : int, default=4
        Multiplier for transition layer hidden dimension

    Examples
    --------
    >>> import torch
    >>> from beignet.nn._pairformer import PairformerBlock
    >>> batch_size, seq_len = 2, 10
    >>> c_s, c_z = 384, 128
    >>> module = PairformerBlock(c_s=c_s, c_z=c_z)
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
        n_head_single: int = 8,
        n_head_pair: int = 4,
        dropout_rate: float = 0.15,
        transition_n: int = 4,
    ):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z

        # Single representation processing
        self.single_row_attention = SingleRowAttention(
            c_s=c_s, n_head=n_head_single, dropout_rate=dropout_rate
        )
        self.single_transition = Transition(c=c_s, n=transition_n)

        # Pair representation processing - triangle operations
        self.triangle_mult_outgoing = TriangleMultiplicationOutgoing(c=c_z)
        self.triangle_mult_incoming = TriangleMultiplicationIncoming(c=c_z)

        self.triangle_attention_starting = TriangleAttentionStartingNode(
            c=c_z, n_head=n_head_pair
        )
        self.triangle_attention_ending = TriangleAttentionEndingNode(
            c=c_z, n_head=n_head_pair
        )

        self.pair_transition = Transition(c=c_z, n=transition_n)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, s_i: Tensor, z_ij: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Forward pass of Pairformer block.

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
        # Process single representation
        s_i = s_i + self.dropout(self.single_row_attention(s_i))
        s_i = s_i + self.dropout(self.single_transition(s_i))

        # Process pair representation with triangle operations
        z_ij = z_ij + self.dropout(self.triangle_mult_outgoing(z_ij))
        z_ij = z_ij + self.dropout(self.triangle_mult_incoming(z_ij))
        z_ij = z_ij + self.dropout(self.triangle_attention_starting(z_ij))
        z_ij = z_ij + self.dropout(self.triangle_attention_ending(z_ij))
        z_ij = z_ij + self.dropout(self.pair_transition(z_ij))

        return s_i, z_ij


class Pairformer(nn.Module):
    r"""
    Pairformer module from AlphaFold 3.

    This is the main trunk processing module that replaces the Evoformer
    from AlphaFold 2. It processes single and pair representations through
    multiple blocks of triangle operations, attention, and transitions.

    Unlike the Evoformer, it does not process MSA representations directly,
    making it simpler and more general for different types of molecules.

    Parameters
    ----------
    n_block : int, default=48
        Number of Pairformer blocks
    c_s : int, default=384
        Channel dimension for single representation
    c_z : int, default=128
        Channel dimension for pair representation
    n_head_single : int, default=8
        Number of attention heads for single representation
    n_head_pair : int, default=4
        Number of attention heads for pair representation
    dropout_rate : float, default=0.15
        Dropout rate
    transition_n : int, default=4
        Multiplier for transition layer hidden dimension

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import Pairformer
    >>> batch_size, seq_len = 2, 10
    >>> n_block, c_s, c_z = 4, 384, 128
    >>> module = Pairformer(n_block=n_block, c_s=c_s, c_z=c_z)
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
    """

    def __init__(
        self,
        n_block: int = 48,
        c_s: int = 384,
        c_z: int = 128,
        n_head_single: int = 8,
        n_head_pair: int = 4,
        dropout_rate: float = 0.15,
        transition_n: int = 4,
    ):
        super().__init__()

        self.n_block = n_block
        self.c_s = c_s
        self.c_z = c_z

        # Create n_block Pairformer blocks (each with its own parameters)
        self.blocks = nn.ModuleList(
            [
                PairformerBlock(
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
        Forward pass of Pairformer.

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

        # Process through all Pairformer blocks
        for block in self.blocks:
            s_i, z_ij = block(s_i, z_ij)

        return s_i, z_ij
