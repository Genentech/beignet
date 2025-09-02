import torch.nn as nn
from torch import Tensor

from ._attention_pair_bias import AttentionPairBias
from ._conditioned_transition_block import ConditionedTransitionBlock


class DiffusionTransformer(nn.Module):
    r"""
    Diffusion Transformer from AlphaFold 3 Algorithm 23.

    This implements a transformer block for diffusion models that alternates between
    AttentionPairBias and ConditionedTransitionBlock operations. The module processes
    single representations {ai} conditioned on {si}, pair representations {zij},
    and bias terms {βij}.

    Parameters
    ----------
    c_a : int
        Channel dimension for single representation 'a'
    c_s : int
        Channel dimension for conditioning signal 's'
    c_z : int
        Channel dimension for pair representation 'z'
    n_head : int
        Number of attention heads
    n_block : int
        Number of transformer blocks
    n : int, default=2
        Expansion factor for ConditionedTransitionBlock

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import DiffusionTransformer
    >>> batch_size, seq_len, c_a, c_s, c_z = 2, 32, 256, 384, 128
    >>> n_head, n_block = 16, 4
    >>> module = DiffusionTransformer(
    ...     c_a=c_a, c_s=c_s, c_z=c_z,
    ...     n_head=n_head, n_block=n_block
    ... )
    >>> a = torch.randn(batch_size, seq_len, c_a)
    >>> s = torch.randn(batch_size, seq_len, c_s)
    >>> z = torch.randn(batch_size, seq_len, seq_len, c_z)
    >>> beta = torch.randn(batch_size, seq_len, seq_len, n_head)
    >>> a_out = module(a, s, z, beta)
    >>> a_out.shape
    torch.Size([2, 32, 256])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 23: Diffusion Transformer
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        n_head: int,
        n_block: int,
        n: int = 2,
    ):
        super().__init__()

        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.n_head = n_head
        self.n_block = n_block
        self.n = n

        # Create n_block pairs of (AttentionPairBias, ConditionedTransitionBlock)
        self.blocks = nn.ModuleList()
        for _ in range(n_block):
            # Each block contains:
            # 1. AttentionPairBias for step 2
            # 2. ConditionedTransitionBlock for step 3
            block = nn.ModuleDict(
                {
                    "attention": AttentionPairBias(
                        c_a=c_a, c_s=c_s, c_z=c_z, n_head=n_head
                    ),
                    "transition": ConditionedTransitionBlock(c=c_a, c_s=c_s, n=n),
                }
            )
            self.blocks.append(block)

    def forward(self, a: Tensor, s: Tensor, z: Tensor, beta: Tensor) -> Tensor:
        r"""
        Forward pass of Diffusion Transformer.

        Parameters
        ----------
        a : Tensor, shape=(batch_size, seq_len, c_a)
            Single representations
        s : Tensor, shape=(batch_size, seq_len, c_s)
            Conditioning signal
        z : Tensor, shape=(batch_size, seq_len, seq_len, c_z)
            Pair representations
        beta : Tensor, shape=(batch_size, seq_len, seq_len, n_head)
            Bias terms for attention

        Returns
        -------
        a_out : Tensor, shape=(batch_size, seq_len, c_a)
            Updated single representations after diffusion transformer
        """
        # Algorithm 23: for all n ∈ [1, ..., N_block] do
        for block in self.blocks:
            # Algorithm 23 Step 2: {bi} = AttentionPairBias({ai}, {si}, {zij}, {βij}, N_head)
            b = block["attention"](a, s, z, beta)

            # Algorithm 23 Step 3: ai ← bi + ConditionedTransitionBlock(ai, si)
            a = b + block["transition"](a, s)

        # Algorithm 23 Step 5: return {ai}
        return a
