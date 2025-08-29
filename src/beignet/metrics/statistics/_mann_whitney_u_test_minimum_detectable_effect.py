from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import mann_whitney_u_test_minimum_detectable_effect


class MannWhitneyUTestMinimumDetectableEffect(Metric):
    r"""
    Compute minimum detectable effect for Mann-Whitney U test.

    This metric calculates the minimum detectable effect size for Mann-Whitney U test
    given desired power, significance level, and sample sizes. It follows TorchMetrics
    conventions for stateful metric computation.

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
    >>> from beignet.metrics import MannWhitneyUTestMinimumDetectableEffect
    >>> metric = MannWhitneyUTestMinimumDetectableEffect()
    >>> sample_size1 = torch.tensor(30)
    >>> sample_size2 = torch.tensor(25)
    >>> metric.update(sample_size1, sample_size2)
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
        self.add_state("sample_size1_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size2_values", default=[], dist_reduce_fx="cat")

    def update(self, sample_size1: Tensor, sample_size2: Tensor) -> None:
        """
        Update the metric state with sample sizes.

        Parameters
        ----------
        sample_size1 : Tensor
            Sample size for first group.
        sample_size2 : Tensor
            Sample size for second group.
        """
        self.sample_size1_values.append(sample_size1)
        self.sample_size2_values.append(sample_size2)

    def compute(self) -> Tensor:
        """
        Compute minimum detectable effect based on stored parameters.

        Returns
        -------
        Tensor
            The computed minimum detectable effect size.
        """
        if not self.sample_size1_values or not self.sample_size2_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        sample_size1 = self.sample_size1_values[-1]
        sample_size2 = self.sample_size2_values[-1]

        # Use functional implementation
        return mann_whitney_u_test_minimum_detectable_effect(
            sample_size1,
            sample_size2,
            power=self.power,
            alpha=self.alpha,
            alternative=self.alternative,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.sample_size1_values = []
        self.sample_size2_values = []

    def plot(
        self,
        plot_type: str = "effect_size_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Mann-Whitney U test minimum detectable effect visualization.

        Parameters
        ----------
        plot_type : str, default="effect_size_curve"
            Type of plot to create. Currently only "effect_size_curve" is supported.
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

        if not self.sample_size1_values or not self.sample_size2_values:
            raise RuntimeError(
                "No values have been added to the metric. Call update() first.",
            )

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "effect_size_curve":
            # Get current parameters
            current_n1 = float(self.sample_size1_values[-1])
            current_n2 = float(self.sample_size2_values[-1])

            # Create range of sample sizes for visualization
            sample_sizes = np.linspace(10, 100, 50)
            effect_sizes = []

            for n in sample_sizes:
                es = mann_whitney_u_test_minimum_detectable_effect(
                    torch.tensor(n),
                    torch.tensor(n),  # Equal sample sizes for simplicity
                    power=self.power,
                    alpha=self.alpha,
                    alternative=self.alternative,
                )
                effect_sizes.append(float(es))

            ax.plot(sample_sizes, effect_sizes, **kwargs)

            # Mark current point
            current_es = float(self.compute())
            ax.plot(
                (current_n1 + current_n2) / 2,
                current_es,
                "ro",
                markersize=8,
                label=f"Current (N1={current_n1}, N2={current_n2}, ES={current_es:.3f})",
            )

            # Add reference lines
            ax.axhline(
                y=0.5,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Medium Effect",
            )
            ax.axhline(
                y=0.8,
                color="gray",
                linestyle=":",
                alpha=0.5,
                label="Large Effect",
            )

            ax.set_xlabel("Sample Size (per group)")
            ax.set_ylabel("Minimum Detectable Effect Size")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"Mann-Whitney U Test Minimum Detectable Effect (Power={self.power}, α={self.alpha})"
        else:
            raise ValueError(f"plot_type must be 'effect_size_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha}, alternative='{self.alternative}')"
