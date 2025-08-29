from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import mann_whitney_u_test_sample_size


class MannWhitneyUTestSampleSize(Metric):
    r"""
    Compute required sample size for Mann-Whitney U test.

    This metric calculates the required sample size for Mann-Whitney U test to achieve
    a desired statistical power. It follows TorchMetrics conventions for stateful
    metric computation.

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
    >>> from beignet.metrics import MannWhitneyUTestSampleSize
    >>> metric = MannWhitneyUTestSampleSize()
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

        # State for storing test parameters
        self.add_state("auc_values", default=[], dist_reduce_fx="cat")
        self.add_state("ratio_values", default=[], dist_reduce_fx="cat")

    def update(self, auc: Tensor, ratio: Tensor) -> None:
        """
        Update the metric state with test parameters.

        Parameters
        ----------
        auc : Tensor
            Area under the curve (effect size measure).
        ratio : Tensor
            Sample size ratio between groups.
        """
        self.auc_values.append(torch.atleast_1d(auc))
        self.ratio_values.append(torch.atleast_1d(ratio))

    def compute(self) -> Tensor:
        """
        Compute required sample size based on stored parameters.

        Returns
        -------
        Tensor
            The computed required sample size.
        """
        if not self.auc_values or not self.ratio_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        auc = self.auc_values[-1]
        ratio = self.ratio_values[-1]

        # Use functional implementation
        return mann_whitney_u_test_sample_size(
            auc,
            ratio=ratio,
            power=self.power,
            alpha=self.alpha,
            alternative=self.alternative,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.auc_values = []
        self.ratio_values = []

    def plot(
        self,
        plot_type: str = "sample_size_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Mann-Whitney U test sample size visualization.

        Parameters
        ----------
        plot_type : str, default="sample_size_curve"
            Type of plot to create. Currently only "sample_size_curve" is supported.
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
            # Get current parameters
            current_effect = float(self.effect_size_values[-1])

            # Create range of effect sizes
            effect_sizes = np.linspace(0.1, 1.5, 100)
            sample_sizes = []

            for es in effect_sizes:
                n = mann_whitney_u_test_sample_size(
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
                title = f"Mann-Whitney U Test Sample Size (Power={self.power}, α={self.alpha})"
        else:
            raise ValueError(f"plot_type must be 'sample_size_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha}, alternative='{self.alternative}')"
