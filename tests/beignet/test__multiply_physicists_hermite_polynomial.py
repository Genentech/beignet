import pytest
import torch

import beignet
from beignet import default_dtype_manager


@pytest.mark.parametrize("dtype", [torch.float64])
def test_multiply_physicists_hermite_polynomial(dtype):
    with default_dtype_manager(dtype):
        for i in range(5):
            input = torch.linspace(-3, 3, 100)

            val1 = beignet.evaluate_physicists_hermite_polynomial(
                input,
                torch.tensor([0.0] * i + [1.0]),
            )

            for j in range(5):
                val2 = beignet.evaluate_physicists_hermite_polynomial(
                    input,
                    torch.tensor([0.0] * j + [1.0]),
                )

                torch.testing.assert_close(
                    beignet.evaluate_physicists_hermite_polynomial(
                        input,
                        beignet.multiply_physicists_hermite_polynomial(
                            torch.tensor([0.0] * i + [1.0]),
                            torch.tensor([0.0] * j + [1.0]),
                        ),
                    ),
                    val1 * val2,
                )
