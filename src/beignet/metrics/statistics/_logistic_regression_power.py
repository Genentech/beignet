from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import logistic_regression_power


class LogisticRegressionPower(Metric):
    r"""
    Compute statistical power for logistic regression.

    This metric calculates the statistical power for logistic regression analysis.
    It follows TorchMetrics conventions for stateful metric computation.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import LogisticRegressionPower
    >>> metric = LogisticRegressionPower()
    >>> odds_ratio = torch.tensor(2.0)
    >>> sample_size = torch.tensor(100)
    >>> baseline_prob = torch.tensor(0.3)
    >>> metric.update(odds_ratio, sample_size, baseline_prob)
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
        self.add_state("odds_ratio_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size_values", default=[], dist_reduce_fx="cat")
        self.add_state("baseline_prob_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        odds_ratio: Tensor,
        sample_size: Tensor,
        baseline_prob: Tensor,
    ) -> None:
        """
        Update the metric state with logistic regression parameters.

        Parameters
        ----------
        odds_ratio : Tensor
            Expected odds ratio.
        sample_size : Tensor
            Total sample size.
        baseline_prob : Tensor
            Baseline probability of outcome.
        """
        self.odds_ratio_values.append(odds_ratio)
        self.sample_size_values.append(sample_size)
        self.baseline_prob_values.append(baseline_prob)

    def compute(self) -> Tensor:
        """
        Compute statistical power based on stored parameters.

        Returns
        -------
        Tensor
            The computed statistical power.
        """
        if (
            not self.odds_ratio_values
            or not self.sample_size_values
            or not self.baseline_prob_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        odds_ratio = self.odds_ratio_values[-1]
        sample_size = self.sample_size_values[-1]
        baseline_prob = self.baseline_prob_values[-1]

        # Use functional implementation
        return logistic_regression_power(
            odds_ratio,
            sample_size,
            baseline_prob,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.odds_ratio_values = []
        self.sample_size_values = []
        self.baseline_prob_values = []

    def plot(
        self,
        plot_type: str = "power_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot logistic regression power visualization.

        Parameters
        ----------
        plot_type : str, default="power_curve"
            Type of plot to create. Currently only "power_curve" is supported.
        ax : matplotlib Axes, optional
            Axes object to plot on. If None, creates new figure.
        title : str, optional
            Plot title. If None, generates automatic title.
        **kwargs : dict
            Additional keyword arguments passed to matplotlib plot function.

        Returns
        -------
        fig : matplotlib Figure
            The figure object containing the plot.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if (
            not self.odds_ratio_values
            or not self.sample_size_values
            or not self.baseline_prob_values
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
            # Get current parameters
            current_or = float(self.odds_ratio_values[-1])
            current_n = float(self.sample_size_values[-1])
            current_prob = float(self.baseline_prob_values[-1])

            # Create range of odds ratios
            odds_ratios = np.linspace(1.2, 4.0, 100)
            powers = []

            for or_val in odds_ratios:
                power = logistic_regression_power(
                    torch.tensor(or_val),
                    torch.tensor(current_n),
                    torch.tensor(current_prob),
                    alpha=self.alpha,
                )
                powers.append(float(power))

            ax.plot(odds_ratios, powers, **kwargs)

            # Mark current point
            current_power = float(self.compute())
            ax.plot(
                current_or,
                current_power,
                "ro",
                markersize=8,
                label=f"Current (OR={current_or:.2f}, Power={current_power:.3f})",
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

            ax.set_xlabel("Odds Ratio")
            ax.set_ylabel("Statistical Power")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"Logistic Regression Power Analysis (α={self.alpha}, N={current_n})"
        else:
            raise ValueError(f"plot_type must be 'power_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"
