import torch
import torch.nn as nn
from torch import Tensor

from ._adaptive_layer_norm import AdaptiveLayerNorm


class ConditionedTransitionBlock(nn.Module):
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
    >>> from beignet.nn import ConditionedTransitionBlock
    >>> batch_size, seq_len, c, c_s = 2, 10, 256, 384
    >>> module = ConditionedTransitionBlock(c=c, c_s=c_s, n=4)
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
