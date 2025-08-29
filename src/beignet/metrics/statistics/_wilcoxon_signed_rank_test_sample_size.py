from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import wilcoxon_signed_rank_test_sample_size


class WilcoxonSignedRankTestSampleSize(Metric):
    r"""
    Compute required sample size for Wilcoxon signed-rank test.

    This metric calculates the required sample size for Wilcoxon signed-rank test
    to achieve a desired statistical power. It follows TorchMetrics conventions
    for stateful metric computation.

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
    >>> from beignet.metrics import WilcoxonSignedRankTestSampleSize
    >>> metric = WilcoxonSignedRankTestSampleSize()
    >>> effect_size = torch.tensor(0.5)
    >>> metric.update(effect_size)
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

        # State for storing effect size parameters
        self.add_state("effect_size_values", default=[], dist_reduce_fx="cat")

    def update(self, effect_size: Tensor) -> None:
        """
        Update the metric state with effect size.

        Parameters
        ----------
        effect_size : Tensor
            Effect size measure.
        """
        self.effect_size_values.append(effect_size)

    def compute(self) -> Tensor:
        """
        Compute required sample size based on stored parameters.

        Returns
        -------
        Tensor
            The computed required sample size.
        """
        if not self.effect_size_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        effect_size = self.effect_size_values[-1]

        # Use functional implementation
        return wilcoxon_signed_rank_test_sample_size(
            effect_size,
            power=self.power,
            alpha=self.alpha,
            alternative=self.alternative,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.effect_size_values = []

    def plot(
        self,
        plot_type: str = "sample_size_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Wilcoxon signed-rank test sample size visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.effect_size_values:
            raise RuntimeError(
                "No values have been added to the metric. Call update() first.",
            )

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "sample_size_curve":
            current_effect = float(self.effect_size_values[-1])

            # Create range of effect sizes
            effect_sizes = np.linspace(0.1, 1.5, 100)
            sample_sizes = []

            for es in effect_sizes:
                n = wilcoxon_signed_rank_test_sample_size(
                    torch.tensor(es),
                    power=self.power,
                    alpha=self.alpha,
                    alternative=self.alternative,
                )
                sample_sizes.append(float(n))

            ax.plot(effect_sizes, sample_sizes, **kwargs)

            # Mark current point
            current_n = float(self.compute())
            ax.plot(
                current_effect,
                current_n,
                "ro",
                markersize=8,
                label=f"Current (ES={current_effect:.2f}, N={current_n:.0f})",
            )

            # Add reference lines for common sample sizes
            ax.axhline(y=30, color="gray", linestyle="--", alpha=0.5, label="N = 30")
            ax.axhline(y=50, color="gray", linestyle=":", alpha=0.5, label="N = 50")

            ax.set_xlabel("Effect Size")
            ax.set_ylabel("Required Sample Size")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"Wilcoxon Signed-Rank Test Sample Size (Power={self.power}, α={self.alpha})"
        else:
            raise ValueError(f"plot_type must be 'sample_size_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha}, alternative='{self.alternative}')"
