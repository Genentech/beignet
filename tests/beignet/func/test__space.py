from typing import Callable

import beignet.func
import hypothesis
import hypothesis.strategies
import torch.testing


def map_product(fn: Callable) -> Callable:
    return torch.vmap(
        torch.vmap(
            fn,
            in_dims=(0, None),
            out_dims=0,
        ),
        in_dims=(None, 0),
        out_dims=0,
    )


@hypothesis.strategies.composite
def _strategy(function):
    dtype = function(
        hypothesis.strategies.sampled_from(
            [
                torch.float32,
                torch.float64,
            ],
        ),
    )

    maximum_size = function(
        hypothesis.strategies.floats(
            min_value=1.0,
            max_value=8.0,
        ),
    )

    particles = function(
        hypothesis.strategies.integers(
            min_value=16,
            max_value=32,
        ),
    )

    spatial_dimension = function(
        hypothesis.strategies.integers(
            min_value=1,
            max_value=3,
        ),
    )

    return (
        dtype,
        torch.rand([particles, spatial_dimension], dtype=dtype),
        particles,
        torch.rand([spatial_dimension], dtype=dtype) * maximum_size,
        spatial_dimension,
    )


@hypothesis.given(_strategy())
@hypothesis.settings(deadline=None)
def test_space(data):
    dtype, input, particles, size, spatial_dimension = data

    displacement_fn, shift_fn = beignet.func.space(size, parallelepiped=False)

    (
        parallelepiped_displacement_fn,
        parallelepiped_shift_fn,
    ) = beignet.func.space(
        torch.diag(size),
    )

    standardized_input = input * size

    torch.testing.assert_close(
        map_product(
            displacement_fn,
        )(
            standardized_input,
            standardized_input,
        ),
        map_product(
            parallelepiped_displacement_fn,
        )(
            input,
            input,
        ),
    )

    displacement = torch.randn([particles, spatial_dimension], dtype=dtype)

    torch.testing.assert_close(
        shift_fn(standardized_input, displacement),
        parallelepiped_shift_fn(input, displacement) * size,
    )
