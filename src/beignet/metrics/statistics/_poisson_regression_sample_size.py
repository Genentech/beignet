from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import poisson_regression_sample_size


class PoissonRegressionSampleSize(Metric):
    r"""
    Compute required sample size for Poisson regression.

    This metric calculates the required sample size for Poisson regression
    to achieve a desired statistical power. It follows TorchMetrics conventions
    for stateful metric computation.

    Parameters
    ----------
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import PoissonRegressionSampleSize
    >>> metric = PoissonRegressionSampleSize()
    >>> rate_ratio = torch.tensor(1.5)
    >>> baseline_rate = torch.tensor(0.1)
    >>> metric.update(rate_ratio, baseline_rate)
    >>> metric.compute()
    tensor(...)
    """

    def __init__(
        self,
        power: float = 0.8,
        alpha: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.power = power
        self.alpha = alpha

        # Validate parameters
        if not 0 < power < 1:
            raise ValueError(f"power must be between 0 and 1, got {power}")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        # State for storing analysis parameters
        self.add_state("rate_ratio_values", default=[], dist_reduce_fx="cat")
        self.add_state("baseline_rate_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        rate_ratio: Tensor,
        baseline_rate: Tensor,
    ) -> None:
        """
        Update the metric state with regression parameters.

        Parameters
        ----------
        rate_ratio : Tensor
            Expected rate ratio (incidence rate ratio).
        baseline_rate : Tensor
            Baseline incidence rate.
        """
        self.rate_ratio_values.append(rate_ratio)
        self.baseline_rate_values.append(baseline_rate)

    def compute(self) -> Tensor:
        """
        Compute required sample size based on stored parameters.

        Returns
        -------
        Tensor
            The computed required sample size.
        """
        if not self.rate_ratio_values or not self.baseline_rate_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        rate_ratio = self.rate_ratio_values[-1]
        baseline_rate = self.baseline_rate_values[-1]

        # Use functional implementation
        return poisson_regression_sample_size(
            rate_ratio,
            baseline_rate,
            power=self.power,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.rate_ratio_values = []
        self.baseline_rate_values = []

    def plot(
        self,
        plot_type: str = "sample_size_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Poisson regression sample size visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.rate_ratio_values or not self.baseline_rate_values:
            raise RuntimeError(
                "No values have been added to the metric. Call update() first.",
            )

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "sample_size_curve":
            current_rr = float(self.rate_ratio_values[-1])
            current_baseline = float(self.baseline_rate_values[-1])

            # Create range of rate ratios
            rate_ratios = np.linspace(1.1, 3.0, 100)
            sample_sizes = []

            for rr in rate_ratios:
                n = poisson_regression_sample_size(
                    torch.tensor(rr),
                    torch.tensor(current_baseline),
                    power=self.power,
                    alpha=self.alpha,
                )
                sample_sizes.append(float(n))

            ax.plot(rate_ratios, sample_sizes, **kwargs)

            # Mark current point
            current_n = float(self.compute())
            ax.plot(
                current_rr,
                current_n,
                "ro",
                markersize=8,
                label=f"Current (RR={current_rr:.1f}, N={current_n:.0f})",
            )

            # Add reference lines
            ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="N = 100")
            ax.axhline(y=500, color="gray", linestyle=":", alpha=0.5, label="N = 500")

            ax.set_xlabel("Rate Ratio")
            ax.set_ylabel("Required Sample Size")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"Poisson Regression Sample Size (Power={self.power}, α={self.alpha})"
        else:
            raise ValueError(f"plot_type must be 'sample_size_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha})"
