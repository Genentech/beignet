from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import interrupted_time_series_power


class InterruptedTimeSeriesPower(Metric):
    r"""
    Compute statistical power for interrupted time series analysis.

    This metric calculates the statistical power for interrupted time series
    analysis to detect a specified effect size. It follows TorchMetrics
    conventions for stateful metric computation.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import InterruptedTimeSeriesPower
    >>> metric = InterruptedTimeSeriesPower()
    >>> effect_size = torch.tensor(0.5)
    >>> sample_size = torch.tensor(50)
    >>> pre_periods = torch.tensor(20)
    >>> post_periods = torch.tensor(30)
    >>> metric.update(effect_size, sample_size, pre_periods, post_periods)
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
        self.add_state("effect_size_values", default=[], dist_reduce_fx="cat")
        self.add_state("n_time_points_values", default=[], dist_reduce_fx="cat")
        self.add_state("n_pre_intervention_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        effect_size: Tensor,
        n_time_points: Tensor,
        n_pre_intervention: Tensor,
    ) -> None:
        """
        Update the metric state with analysis parameters.

        Parameters
        ----------
        effect_size : Tensor
            Effect size of the intervention.
        n_time_points : Tensor
            Total number of time points.
        n_pre_intervention : Tensor
            Number of pre-intervention periods.
        """
        self.effect_size_values.append(torch.atleast_1d(effect_size))
        self.n_time_points_values.append(torch.atleast_1d(n_time_points))
        self.n_pre_intervention_values.append(torch.atleast_1d(n_pre_intervention))

    def compute(self) -> Tensor:
        """
        Compute statistical power based on stored parameters.

        Returns
        -------
        Tensor
            The computed statistical power.
        """
        if (
            not self.effect_size_values
            or not self.n_time_points_values
            or not self.n_pre_intervention_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        effect_size = self.effect_size_values[-1]
        n_time_points = self.n_time_points_values[-1]
        n_pre_intervention = self.n_pre_intervention_values[-1]

        # Use functional implementation
        return interrupted_time_series_power(
            effect_size,
            n_time_points,
            n_pre_intervention,
            autocorrelation=torch.tensor(0.0),
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.effect_size_values = []
        self.n_time_points_values = []
        self.n_pre_intervention_values = []

    def plot(
        self,
        plot_type: str = "power_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot interrupted time series power visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if (
            not self.effect_size_values
            or not self.sample_size_values
            or not self.pre_periods_values
            or not self.post_periods_values
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
            current_effect = float(self.effect_size_values[-1])
            current_n = float(self.sample_size_values[-1])
            current_pre = float(self.pre_periods_values[-1])
            current_post = float(self.post_periods_values[-1])

            # Create range of effect sizes
            effect_sizes = np.linspace(0.1, 1.5, 100)
            powers = []

            for es in effect_sizes:
                power = interrupted_time_series_power(
                    torch.tensor(es),
                    torch.tensor(current_n),
                    torch.tensor(current_pre),
                    torch.tensor(current_post),
                    alpha=self.alpha,
                )
                powers.append(float(power))

            ax.plot(effect_sizes, powers, **kwargs)

            # Mark current point
            current_power = float(self.compute())
            ax.plot(
                current_effect,
                current_power,
                "ro",
                markersize=8,
                label=f"Current (ES={current_effect:.2f}, Power={current_power:.3f})",
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

            ax.set_xlabel("Effect Size")
            ax.set_ylabel("Statistical Power")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"Interrupted Time Series Power (α={self.alpha}, N={current_n})"
        else:
            raise ValueError(f"plot_type must be 'power_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"
