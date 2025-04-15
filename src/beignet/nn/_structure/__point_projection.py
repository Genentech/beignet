from torch.nn import Linear, Module


class PointProjection(Module):
    def __init__(
        self,
        c_hidden: int,
        num_points: int,
        no_heads: int,
        is_multimer: bool,
        return_local_points: bool = False,
    ):
        super().__init__()

        self.return_local_points = return_local_points

        self.no_heads = no_heads

        self.num_points = num_points

        self.is_multimer = is_multimer

        self.linear = Linear(c_hidden, no_heads * 3 * num_points)
