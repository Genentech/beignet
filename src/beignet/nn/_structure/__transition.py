from torch import Tensor
from torch.nn import Linear, Module, ReLU


class _Transition(Module):
    def __init__(self, c):
        super().__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = ReLU()

    def forward(self, input: Tensor) -> Tensor:
        s_initial = input
        input = self.linear_1(input)
        input = self.relu(input)
        input = self.linear_2(input)
        input = self.relu(input)
        input = self.linear_3(input)
        input = input + s_initial
        return input
