from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import analysis_of_variance_minimum_detectable_effect


class AnalysisOfVarianceMinimumDetectableEffect(Metric):
    r"""
    Compute minimum detectable effect for Analysis of Variance (ANOVA).

    This metric calculates the minimum detectable effect for ANOVA given
    sample size and power. It follows TorchMetrics conventions for
    stateful metric computation.

    Parameters
    ----------
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import AnalysisOfVarianceMinimumDetectableEffect
    >>> metric = AnalysisOfVarianceMinimumDetectableEffect()
    >>> sample_size = torch.tensor(60)
    >>> groups = torch.tensor(3)
    >>> metric.update(sample_size, groups)
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
        self.add_state("sample_size_values", default=[], dist_reduce_fx="cat")
        self.add_state("groups_values", default=[], dist_reduce_fx="cat")

    def update(self, sample_size: Tensor, groups: Tensor) -> None:
        """
        Update the metric state with ANOVA parameters.

        Parameters
        ----------
        sample_size : Tensor
            Total sample size.
        groups : Tensor
            Number of groups.
        """
        self.sample_size_values.append(sample_size)
        self.groups_values.append(groups)

    def compute(self) -> Tensor:
        """
        Compute minimum detectable effect based on stored parameters.

        Returns
        -------
        Tensor
            The computed minimum detectable effect.
        """
        if not self.sample_size_values or not self.groups_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        sample_size = self.sample_size_values[-1]
        groups = self.groups_values[-1]

        # Use functional implementation
        return analysis_of_variance_minimum_detectable_effect(
            sample_size,
            groups,
            power=self.power,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.sample_size_values = []
        self.groups_values = []

    def plot(
        self,
        plot_type: str = "effect_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot ANOVA minimum detectable effect visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.sample_size_values or not self.groups_values:
            raise RuntimeError(
                "No values have been added to the metric. Call update() first.",
            )

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "effect_curve":
            current_n = float(self.sample_size_values[-1])
            current_groups = float(self.groups_values[-1])

            # Create range of sample sizes
            sample_sizes = np.arange(20, 200, 5)
            effects = []

            for n in sample_sizes:
                effect = analysis_of_variance_minimum_detectable_effect(
                    torch.tensor(n),
                    torch.tensor(current_groups),
                    power=self.power,
                    alpha=self.alpha,
                )
                effects.append(float(effect))

            ax.plot(sample_sizes, effects, **kwargs)

            # Mark current point
            current_effect = float(self.compute())
            ax.plot(
                current_n,
                current_effect,
                "ro",
                markersize=8,
                label=f"Current (N={current_n:.0f}, MDE={current_effect:.3f})",
            )

            # Add reference lines for effect sizes
            ax.axhline(
                y=0.25, color="gray", linestyle="--", alpha=0.5, label="f = 0.25"
            )
            ax.axhline(y=0.40, color="gray", linestyle=":", alpha=0.5, label="f = 0.40")

            ax.set_xlabel("Sample Size")
            ax.set_ylabel("Minimum Detectable Effect Size (f)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"ANOVA MDE (Power={self.power}, α={self.alpha}, Groups={current_groups})"
        else:
            raise ValueError(f"plot_type must be 'effect_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha})"
