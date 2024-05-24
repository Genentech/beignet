from typing import Callable

import torch
from torch import Tensor

from .__npt_nose_hoover_chain_state import _NPTNoseHooverChainState
from .__volume_metric import _volume_metric


def _npt_box_info(
    state: _NPTNoseHooverChainState,
) -> (float, Callable[[float], float]):
    spatial_dimension = state.positions.shape[1]

    reference_box = state.reference_box

    v_0 = _volume_metric(spatial_dimension, reference_box)

    v = torch.multiply(
        v_0,
        torch.exp(
            torch.multiply(
                state.current_box_positions,
                spatial_dimension,
            ),
        ),
    )

    def fn(_v: Tensor) -> Tensor:
        return torch.pow(
            torch.divide(
                _v,
                v_0,
            ),
            torch.multiply(
                reference_box,
                1 / spatial_dimension,
            ),
        )

    return v, fn
