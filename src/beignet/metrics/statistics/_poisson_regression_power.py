from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import poisson_regression_power


class PoissonRegressionPower(Metric):
    r"""
    Compute statistical power for Poisson regression.

    This metric calculates the statistical power for Poisson regression
    to detect a specified rate ratio. It follows TorchMetrics conventions
    for stateful metric computation.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import PoissonRegressionPower
    >>> metric = PoissonRegressionPower()
    >>> rate_ratio = torch.tensor(1.5)
    >>> sample_size = torch.tensor(200)
    >>> baseline_rate = torch.tensor(0.1)
    >>> metric.update(rate_ratio, sample_size, baseline_rate)
    >>> metric.compute()
    tensor(...)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.alpha = alpha

        # Validate parameters
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        # State for storing analysis parameters
        self.add_state("rate_ratio_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size_values", default=[], dist_reduce_fx="cat")
        self.add_state("baseline_rate_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        rate_ratio: Tensor,
        sample_size: Tensor,
        baseline_rate: Tensor,
    ) -> None:
        """
        Update the metric state with regression parameters.

        Parameters
        ----------
        rate_ratio : Tensor
            Expected rate ratio (incidence rate ratio).
        sample_size : Tensor
            Sample size.
        baseline_rate : Tensor
            Baseline incidence rate.
        """
        self.rate_ratio_values.append(rate_ratio)
        self.sample_size_values.append(sample_size)
        self.baseline_rate_values.append(baseline_rate)

    def compute(self) -> Tensor:
        """
        Compute statistical power based on stored parameters.

        Returns
        -------
        Tensor
            The computed statistical power.
        """
        if (
            not self.rate_ratio_values
            or not self.sample_size_values
            or not self.baseline_rate_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        rate_ratio = self.rate_ratio_values[-1]
        sample_size = self.sample_size_values[-1]
        baseline_rate = self.baseline_rate_values[-1]

        # Use functional implementation
        return poisson_regression_power(
            rate_ratio,
            sample_size,
            baseline_rate,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.rate_ratio_values = []
        self.sample_size_values = []
        self.baseline_rate_values = []

    def plot(
        self,
        plot_type: str = "power_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Poisson regression power visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if (
            not self.rate_ratio_values
            or not self.sample_size_values
            or not self.baseline_rate_values
        ):
            raise RuntimeError(
                "No values have been added to the metric. Call update() first.",
            )

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "power_curve":
            current_rr = float(self.rate_ratio_values[-1])
            current_n = float(self.sample_size_values[-1])
            current_baseline = float(self.baseline_rate_values[-1])

            # Create range of rate ratios
            rate_ratios = np.linspace(1.1, 3.0, 100)
            powers = []

            for rr in rate_ratios:
                power = poisson_regression_power(
                    torch.tensor(rr),
                    torch.tensor(current_n),
                    torch.tensor(current_baseline),
                    alpha=self.alpha,
                )
                powers.append(float(power))

            ax.plot(rate_ratios, powers, **kwargs)

            # Mark current point
            current_power = float(self.compute())
            ax.plot(
                current_rr,
                current_power,
                "ro",
                markersize=8,
                label=f"Current (RR={current_rr:.1f}, Power={current_power:.3f})",
            )

            # Add reference lines
            ax.axhline(
                y=0.8,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Power = 0.8",
            )
            ax.axhline(
                y=0.5,
                color="gray",
                linestyle=":",
                alpha=0.5,
                label="Power = 0.5",
            )

            ax.set_xlabel("Rate Ratio")
            ax.set_ylabel("Statistical Power")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"Poisson Regression Power (α={self.alpha}, N={current_n})"
        else:
            raise ValueError(f"plot_type must be 'power_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"
