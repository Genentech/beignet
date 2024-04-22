import torch


def _map_bond(distance_fn):
    def wrapper(start_positions, end_positions):
        batch_size = start_positions.shape[0]
        return torch.stack(
            [
                distance_fn(start_positions[i], end_positions[i])
                for i in range(batch_size)
            ]
        )

    return wrapper
