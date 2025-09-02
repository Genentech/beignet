import torch
import torch.nn as nn
from torch import Tensor

from ._adaptive_layer_norm import AdaptiveLayerNorm
from ._attention_pair_bias import AttentionPairBias


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
                    "transition": _ConditionedTransitionBlock(c=c_a, c_s=c_s, n=n),
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


class _ConditionedTransitionBlock(nn.Module):
    r"""
    Conditioned Transition Block from AlphaFold 3 Algorithm 25.

    This implements a SwiGLU transition block with adaptive layer normalization
    that conditions the transition on an additional signal 's'. This is used
    in AlphaFold 3 for conditioning various components.

    Parameters
    ----------
    c : int
        Channel dimension for input tensor 'a'
    c_s : int
        Channel dimension for conditioning signal 's'
    n : int, default=2
        Expansion factor for the hidden dimension

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import _ConditionedTransitionBlock
    >>> batch_size, seq_len, c, c_s = 2, 10, 256, 384
    >>> module = _ConditionedTransitionBlock(c=c, c_s=c_s, n=4)
    >>> a = torch.randn(batch_size, seq_len, c)
    >>> s = torch.randn(batch_size, seq_len, c_s)
    >>> a_out = module(a, s)
    >>> a_out.shape
    torch.Size([2, 10, 256])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 25: Conditioned Transition Block
    """

    def __init__(self, c: int, c_s: int, n: int = 2):
        super().__init__()

        self.c = c
        self.c_s = c_s
        self.n = n
        self.hidden_dim = n * c

        # Step 1: AdaLN(a, s) - Adaptive Layer Normalization
        self.ada_ln = AdaptiveLayerNorm(c=c, c_s=c_s)

        # Step 2: SwiGLU components
        # swish(LinearNoBias(a)) ⊙ LinearNoBias(a)
        self.linear_swish = nn.Linear(c, self.hidden_dim, bias=False)
        self.linear_gate = nn.Linear(c, self.hidden_dim, bias=False)

        # Step 3: Output projection (from adaLN-Zero [27])
        # sigmoid(Linear(s, biasinit=-2.0)) ⊙ LinearNoBias(b)
        self.linear_s_gate = nn.Linear(c_s, c, bias=True)
        self.linear_output = nn.Linear(self.hidden_dim, c, bias=False)

        # Initialize the gate bias to -2.0 as specified in the algorithm
        with torch.no_grad():
            self.linear_s_gate.bias.fill_(-2.0)

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        r"""
        Forward pass of Conditioned Transition Block.

        Parameters
        ----------
        a : Tensor, shape=(..., c)
            Input tensor
        s : Tensor, shape=(..., c_s)
            Conditioning signal tensor

        Returns
        -------
        a_out : Tensor, shape=(..., c)
            Output tensor after conditioned transition
        """
        return torch.sigmoid(
            self.linear_s_gate(s),
        ) * self.linear_output(
            torch.nn.functional.silu(
                self.linear_swish(
                    self.ada_ln(
                        a,
                        s,
                    ),
                ),
            )
            * self.linear_gate(
                self.ada_ln(
                    a,
                    s,
                ),
            )
        )
