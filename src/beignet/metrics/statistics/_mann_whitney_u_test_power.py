from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import mann_whitney_u_test_power


class MannWhitneyUTestPower(Metric):
    r"""
    Compute statistical power for Mann-Whitney U test.

    This metric calculates the statistical power for Mann-Whitney U test, which is
    a non-parametric test for comparing two independent groups. It follows
    TorchMetrics conventions for stateful metric computation.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Either "two-sided", "greater", or "less".

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import MannWhitneyUTestPower
    >>> metric = MannWhitneyUTestPower()
    >>> effect_size = torch.tensor(0.5)
    >>> sample_size1 = torch.tensor(30)
    >>> sample_size2 = torch.tensor(25)
    >>> metric.update(effect_size, sample_size1, sample_size2)
    >>> metric.compute()
    tensor(...)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.alpha = alpha
        self.alternative = alternative

        # Validate parameters
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        if alternative not in ["two-sided", "greater", "less"]:
            raise ValueError(
                f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}",
            )

        # State for storing power analysis parameters
        self.add_state("effect_size_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size1_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size2_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        effect_size: Tensor,
        sample_size1: Tensor,
        sample_size2: Tensor,
    ) -> None:
        """
        Update the metric state with power analysis parameters.

        Parameters
        ----------
        effect_size : Tensor
            Effect size measure.
        sample_size1 : Tensor
            Sample size for first group.
        sample_size2 : Tensor
            Sample size for second group.
        """
        self.effect_size_values.append(effect_size)
        self.sample_size1_values.append(sample_size1)
        self.sample_size2_values.append(sample_size2)

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
            or not self.sample_size1_values
            or not self.sample_size2_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        effect_size = self.effect_size_values[-1]
        sample_size1 = self.sample_size1_values[-1]
        sample_size2 = self.sample_size2_values[-1]

        # Use functional implementation
        return mann_whitney_u_test_power(
            effect_size,
            sample_size1,
            sample_size2,
            alpha=self.alpha,
            alternative=self.alternative,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.effect_size_values = []
        self.sample_size1_values = []
        self.sample_size2_values = []

    def plot(
        self,
        plot_type: str = "power_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Mann-Whitney U test power visualization.

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
            not self.effect_size_values
            or not self.sample_size1_values
            or not self.sample_size2_values
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
            current_effect = float(self.effect_size_values[-1])
            current_n1 = float(self.sample_size1_values[-1])
            current_n2 = float(self.sample_size2_values[-1])

            # Create range of effect sizes
            effect_sizes = np.linspace(0.1, 1.5, 100)
            powers = []

            for es in effect_sizes:
                power = mann_whitney_u_test_power(
                    torch.tensor(es),
                    torch.tensor(current_n1),
                    torch.tensor(current_n2),
                    alpha=self.alpha,
                    alternative=self.alternative,
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
                y=0.8, color="gray", linestyle="--", alpha=0.5, label="Power = 0.8"
            )
            ax.axhline(
                y=0.5, color="gray", linestyle=":", alpha=0.5, label="Power = 0.5"
            )

            ax.set_xlabel("Effect Size")
            ax.set_ylabel("Statistical Power")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"Mann-Whitney U Test Power Analysis (α={self.alpha}, N1={current_n1}, N2={current_n2})"
        else:
            raise ValueError(f"plot_type must be 'power_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha}, alternative='{self.alternative}')"
