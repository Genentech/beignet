import torch
import torch.nn as nn
from torch import Tensor


class Transition(nn.Module):
    r"""
    Transition layer from AlphaFold 3.

    This implements Algorithm 11 from AlphaFold 3, which is a simple
    transition layer with layer normalization, two linear projections,
    and a SwiGLU activation function for enhanced non-linearity.

    Parameters
    ----------
    c : int, default=128
        Input and output channel dimension
    n : int, default=4
        Expansion factor for the hidden dimension

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import Transition
    >>> batch_size, seq_len, c = 2, 10, 128
    >>> n = 4
    >>> module = Transition(c=c, n=n)
    >>> x = torch.randn(batch_size, seq_len, c)
    >>> x_out = module(x)
    >>> x_out.shape
    torch.Size([2, 10, 128])

    References
    ----------
    .. [1] AlphaFold 3 paper, Algorithm 11: Transition layer
    """

    def __init__(self, c: int = 128, n: int = 4):
        super().__init__()

        self.c = c
        self.n = n
        self.hidden_dim = n * c

        # Layer normalization (step 1)
        self.layer_norm = nn.LayerNorm(c)

        # First linear projection (step 2)
        self.linear_1 = nn.Linear(c, self.hidden_dim, bias=False)

        # Second linear projection (step 3)
        self.linear_2 = nn.Linear(c, self.hidden_dim, bias=False)

        # Final output projection (step 4)
        self.output_linear = nn.Linear(self.hidden_dim, c, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Forward pass of transition layer.

        Parameters
        ----------
        x : Tensor, shape=(..., c)
            Input tensor where c is the channel dimension.

        Returns
        -------
        x : Tensor, shape=(..., c)
            Output tensor after transition layer processing.
        """
        x = self.layer_norm(x)

        x = self.output_linear(
            self.linear_1(x) * torch.sigmoid(self.linear_1(x)) * self.linear_2(x),
        )  # (..., c)

        return x
