from torch import Tensor
from torch.nn import Linear, Module, ReLU


class ResNetBlock(Module):
    def __init__(self, c_hidden):
        super().__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = ReLU()

    def forward(self, a: Tensor) -> Tensor:
        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial
