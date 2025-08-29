from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import proportional_hazards_model_sample_size


class ProportionalHazardsModelSampleSize(Metric):
    r"""
    Compute required sample size for proportional hazards (Cox) regression.

    This metric calculates the required sample size for proportional hazards
    regression to achieve a desired statistical power. It follows TorchMetrics
    conventions for stateful metric computation.

    Parameters
    ----------
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import ProportionalHazardsModelSampleSize
    >>> metric = ProportionalHazardsModelSampleSize()
    >>> hazard_ratio = torch.tensor(1.5)
    >>> event_probability = torch.tensor(0.7)
    >>> metric.update(hazard_ratio, event_probability)
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
        self.add_state("hazard_ratio_values", default=[], dist_reduce_fx="cat")
        self.add_state("event_probability_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        hazard_ratio: Tensor,
        event_probability: Tensor,
    ) -> None:
        """
        Update the metric state with survival analysis parameters.

        Parameters
        ----------
        hazard_ratio : Tensor
            Expected hazard ratio.
        event_probability : Tensor
            Probability of observing the event.
        """
        self.hazard_ratio_values.append(hazard_ratio)
        self.event_probability_values.append(event_probability)

    def compute(self) -> Tensor:
        """
        Compute required sample size based on stored parameters.

        Returns
        -------
        Tensor
            The computed required sample size.
        """
        if not self.hazard_ratio_values or not self.event_probability_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        hazard_ratio = self.hazard_ratio_values[-1]
        event_prob = self.event_probability_values[-1]

        # Use functional implementation
        return proportional_hazards_model_sample_size(
            hazard_ratio,
            event_prob,
            power=self.power,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.hazard_ratio_values = []
        self.event_probability_values = []

    def plot(
        self,
        plot_type: str = "sample_size_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot proportional hazards model sample size visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.hazard_ratio_values or not self.event_probability_values:
            raise RuntimeError(
                "No values have been added to the metric. Call update() first.",
            )

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "sample_size_curve":
            current_hr = float(self.hazard_ratio_values[-1])
            current_event_prob = float(self.event_probability_values[-1])

            # Create range of hazard ratios
            hazard_ratios = np.linspace(1.1, 3.0, 100)
            sample_sizes = []

            for hr in hazard_ratios:
                n = proportional_hazards_model_sample_size(
                    torch.tensor(hr),
                    torch.tensor(current_event_prob),
                    power=self.power,
                    alpha=self.alpha,
                )
                sample_sizes.append(float(n))

            ax.plot(hazard_ratios, sample_sizes, **kwargs)

            # Mark current point
            current_n = float(self.compute())
            ax.plot(
                current_hr,
                current_n,
                "ro",
                markersize=8,
                label=f"Current (HR={current_hr:.1f}, N={current_n:.0f})",
            )

            # Add reference lines
            ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="N = 100")
            ax.axhline(y=300, color="gray", linestyle=":", alpha=0.5, label="N = 300")

            ax.set_xlabel("Hazard Ratio")
            ax.set_ylabel("Required Sample Size")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = (
                    f"Cox Regression Sample Size (Power={self.power}, α={self.alpha})"
                )
        else:
            raise ValueError(f"plot_type must be 'sample_size_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha})"
