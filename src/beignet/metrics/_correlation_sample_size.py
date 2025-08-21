import math
from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


class CorrelationSampleSize(Metric):
    r"""
    Compute required sample size for correlation tests based on expected correlation.

    This metric calculates the required sample size for achieving a target
    statistical power for detecting a correlation of a specified magnitude.
    It follows TorchMetrics conventions for stateful metric computation.

    Parameters
    ----------
    power : float, default=0.8
        Desired statistical power for sample size calculations.
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import CorrelationSampleSize
    >>> metric = CorrelationSampleSize(power=0.9)
    >>> # Expected correlation of 0.3
    >>> metric.update(torch.tensor(0.3))
    >>> metric.compute()
    tensor(...)
    """

    def __init__(
        self,
        power: float = 0.8,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if alternative not in ["two-sided", "greater", "less"]:
            raise ValueError(f"Unknown alternative: {alternative}")
        if not 0 < alpha < 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
        if not 0 < power < 1:
            raise ValueError(f"Power must be between 0 and 1, got {power}")

        self.power = power
        self.alpha = alpha
        self.alternative = alternative

        # State for storing expected correlation values
        self.add_state("r_values", default=[], dist_reduce_fx="cat")

    def update(self, r: Tensor) -> None:
        """
        Update the metric state with expected correlation coefficient.

        Parameters
        ----------
        r : Tensor
            Expected correlation coefficient.
        """
        self.r_values.append(r)

    def compute(self) -> Tensor:
        """
        Compute required sample size based on stored correlation values.

        Returns
        -------
        Tensor
            The required sample size (rounded up to nearest integer).
        """
        if not self.r_values:
            raise RuntimeError("No correlation values have been added to the metric.")

        # Use the most recent value (or could average them)
        r = (
            self.r_values[-1]
            if len(self.r_values) == 1
            else torch.cat(self.r_values, dim=-1).mean()
        )

        # Inline implementation of correlation_sample_size to avoid circular imports

        # Convert inputs to tensors if needed
        r = torch.as_tensor(r)

        # Fisher z-transformation of the correlation
        # z_r = 0.5 * ln((1 + r) / (1 - r))
        epsilon = 1e-7  # Small value to avoid division by zero
        r_clamped = torch.clamp(r, -1 + epsilon, 1 - epsilon)
        z_r = 0.5 * torch.log((1 + r_clamped) / (1 - r_clamped))

        # Standard normal quantiles using erfinv
        sqrt_2 = math.sqrt(2.0)

        if self.alternative == "two-sided":
            z_alpha = (
                torch.erfinv(torch.tensor(1 - self.alpha / 2, dtype=r.dtype)) * sqrt_2
            )
            z_beta = torch.erfinv(torch.tensor(self.power, dtype=r.dtype)) * sqrt_2
        elif self.alternative in ["greater", "less"]:
            z_alpha = torch.erfinv(torch.tensor(1 - self.alpha, dtype=r.dtype)) * sqrt_2
            z_beta = torch.erfinv(torch.tensor(self.power, dtype=r.dtype)) * sqrt_2
        else:
            raise ValueError(f"Unknown alternative: {self.alternative}")

        # Sample size formula for correlation test
        # For Fisher z-transform: SE = 1/sqrt(n-3), so n = ((z_alpha + z_beta) / z_r)^2 + 3
        # This comes from the power calculation: z_r / (1/sqrt(n-3)) = z_alpha + z_beta

        # Avoid division by very small correlations
        z_r_safe = torch.where(torch.abs(z_r) < 1e-6, torch.sign(z_r) * 1e-6, z_r)

        sample_size = ((z_alpha + z_beta) / torch.abs(z_r_safe)) ** 2 + 3

        # Round up to nearest integer
        return torch.ceil(sample_size)

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.r_values = []

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"power={self.power}, "
            f"alpha={self.alpha}, "
            f"alternative={self.alternative!r})"
        )
