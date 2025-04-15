from torch import Tensor
from torch.nn import Dropout, LayerNorm, Module, ModuleList

from .__transition import _Transition


class Transition(Module):
    def __init__(self, c, num_layers, dropout_rate):
        super().__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = ModuleList()

        for _ in range(self.num_layers):
            layer = _Transition(self.c)

            self.layers.append(layer)

        self.dropout = Dropout(self.dropout_rate)

        self.layer_norm = LayerNorm(self.c)

    def forward(self, input: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer(input)

        input = self.dropout(input)
        input = self.layer_norm(input)

        return input
