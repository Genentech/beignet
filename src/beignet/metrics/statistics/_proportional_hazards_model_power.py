from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import proportional_hazards_model_power


class ProportionalHazardsModelPower(Metric):
    r"""
    Compute statistical power for proportional hazards (Cox) regression.

    This metric calculates the statistical power for proportional hazards
    regression to detect a specified hazard ratio. It follows TorchMetrics
    conventions for stateful metric computation.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import ProportionalHazardsModelPower
    >>> metric = ProportionalHazardsModelPower()
    >>> hazard_ratio = torch.tensor(1.5)
    >>> sample_size = torch.tensor(200)
    >>> event_probability = torch.tensor(0.7)
    >>> metric.update(hazard_ratio, sample_size, event_probability)
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
        self.add_state("hazard_ratio_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size_values", default=[], dist_reduce_fx="cat")
        self.add_state("event_probability_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        hazard_ratio: Tensor,
        sample_size: Tensor,
        event_probability: Tensor,
    ) -> None:
        """
        Update the metric state with survival analysis parameters.

        Parameters
        ----------
        hazard_ratio : Tensor
            Expected hazard ratio.
        sample_size : Tensor
            Total sample size.
        event_probability : Tensor
            Probability of observing the event.
        """
        self.hazard_ratio_values.append(hazard_ratio)
        self.sample_size_values.append(sample_size)
        self.event_probability_values.append(event_probability)

    def compute(self) -> Tensor:
        """
        Compute statistical power based on stored parameters.

        Returns
        -------
        Tensor
            The computed statistical power.
        """
        if (
            not self.hazard_ratio_values
            or not self.sample_size_values
            or not self.event_probability_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        hazard_ratio = self.hazard_ratio_values[-1]
        sample_size = self.sample_size_values[-1]
        event_prob = self.event_probability_values[-1]

        # Use functional implementation
        return proportional_hazards_model_power(
            hazard_ratio,
            sample_size,
            event_prob,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.hazard_ratio_values = []
        self.sample_size_values = []
        self.event_probability_values = []

    def plot(
        self,
        plot_type: str = "power_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot proportional hazards model power visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if (
            not self.hazard_ratio_values
            or not self.sample_size_values
            or not self.event_probability_values
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
            current_hr = float(self.hazard_ratio_values[-1])
            current_n = float(self.sample_size_values[-1])
            current_event_prob = float(self.event_probability_values[-1])

            # Create range of hazard ratios
            hazard_ratios = np.linspace(1.1, 3.0, 100)
            powers = []

            for hr in hazard_ratios:
                power = proportional_hazards_model_power(
                    torch.tensor(hr),
                    torch.tensor(current_n),
                    torch.tensor(current_event_prob),
                    alpha=self.alpha,
                )
                powers.append(float(power))

            ax.plot(hazard_ratios, powers, **kwargs)

            # Mark current point
            current_power = float(self.compute())
            ax.plot(
                current_hr,
                current_power,
                "ro",
                markersize=8,
                label=f"Current (HR={current_hr:.1f}, Power={current_power:.3f})",
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

            ax.set_xlabel("Hazard Ratio")
            ax.set_ylabel("Statistical Power")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"Cox Regression Power (α={self.alpha}, N={current_n})"
        else:
            raise ValueError(f"plot_type must be 'power_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"
