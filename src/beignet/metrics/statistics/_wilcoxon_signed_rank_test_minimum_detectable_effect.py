from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import wilcoxon_signed_rank_test_minimum_detectable_effect


class WilcoxonSignedRankTestMinimumDetectableEffect(Metric):
    r"""
    Compute minimum detectable effect for Wilcoxon signed-rank test.

    This metric calculates the minimum detectable effect for Wilcoxon signed-rank
    test given sample size and power. It follows TorchMetrics conventions for
    stateful metric computation.

    Parameters
    ----------
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Either "two-sided", "greater", or "less".

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import WilcoxonSignedRankTestMinimumDetectableEffect
    >>> metric = WilcoxonSignedRankTestMinimumDetectableEffect()
    >>> sample_size = torch.tensor(30)
    >>> metric.update(sample_size)
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

        self.power = power
        self.alpha = alpha
        self.alternative = alternative

        # Validate parameters
        if not 0 < power < 1:
            raise ValueError(f"power must be between 0 and 1, got {power}")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        if alternative not in ["two-sided", "greater", "less"]:
            raise ValueError(
                f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}",
            )

        # State for storing sample size parameters
        self.add_state("sample_size_values", default=[], dist_reduce_fx="cat")

    def update(self, sample_size: Tensor) -> None:
        """
        Update the metric state with sample size.

        Parameters
        ----------
        sample_size : Tensor
            Sample size.
        """
        self.sample_size_values.append(sample_size)

    def compute(self) -> Tensor:
        """
        Compute minimum detectable effect based on stored parameters.

        Returns
        -------
        Tensor
            The computed minimum detectable effect.
        """
        if not self.sample_size_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        sample_size = self.sample_size_values[-1]

        # Use functional implementation
        return wilcoxon_signed_rank_test_minimum_detectable_effect(
            sample_size,
            power=self.power,
            alpha=self.alpha,
            alternative=self.alternative,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.sample_size_values = []

    def plot(
        self,
        plot_type: str = "effect_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Wilcoxon signed-rank test minimum detectable effect visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.sample_size_values:
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

            # Create range of sample sizes
            sample_sizes = np.arange(10, 200, 5)
            effects = []

            for n in sample_sizes:
                effect = wilcoxon_signed_rank_test_minimum_detectable_effect(
                    torch.tensor(n),
                    power=self.power,
                    alpha=self.alpha,
                    alternative=self.alternative,
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

            # Add reference lines
            ax.axhline(
                y=0.5, color="gray", linestyle="--", alpha=0.5, label="Effect = 0.5"
            )
            ax.axhline(
                y=0.8, color="gray", linestyle=":", alpha=0.5, label="Effect = 0.8"
            )

            ax.set_xlabel("Sample Size")
            ax.set_ylabel("Minimum Detectable Effect")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"Wilcoxon Signed-Rank Test MDE (Power={self.power}, α={self.alpha})"
        else:
            raise ValueError(f"plot_type must be 'effect_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha}, alternative='{self.alternative}')"
