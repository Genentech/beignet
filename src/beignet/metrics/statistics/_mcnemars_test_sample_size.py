from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import mcnemars_test_sample_size


class McnemarsTestSampleSize(Metric):
    r"""
    Compute required sample size for McNemar's test.

    This metric calculates the required sample size for McNemar's test to achieve
    a desired statistical power. It follows TorchMetrics conventions for
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
    >>> from beignet.metrics import McnemarsTestSampleSize
    >>> metric = McnemarsTestSampleSize()
    >>> effect_size = torch.tensor(0.1)
    >>> discordant_proportion = torch.tensor(0.2)
    >>> metric.update(effect_size, discordant_proportion)
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
        self.add_state("discordant_proportion_values", default=[], dist_reduce_fx="cat")

    def update(self, discordant_proportion: Tensor) -> None:
        """
        Update the metric state with test parameters.

        Parameters
        ----------
        discordant_proportion : Tensor
            Total proportion of discordant pairs.
        """
        self.discordant_proportion_values.append(
            torch.atleast_1d(discordant_proportion),
        )

    def compute(self) -> Tensor:
        """
        Compute required sample size based on stored parameters.

        Returns
        -------
        Tensor
            The computed required sample size.
        """
        if not self.discordant_proportion_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        discordant_proportion = self.discordant_proportion_values[-1]

        # Split discordant proportion equally for balanced case
        p01 = p10 = discordant_proportion / 2.0

        # Use functional implementation
        return mcnemars_test_sample_size(
            p01,
            p10,
            power=self.power,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.discordant_proportion_values = []

    def plot(
        self,
        plot_type: str = "sample_size_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot McNemar's test sample size visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.effect_size_values or not self.discordant_proportion_values:
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
            current_discord = float(self.discordant_proportion_values[-1])

            # Create range of effect sizes
            effect_sizes = np.linspace(0.05, 0.5, 100)
            sample_sizes = []

            for es in effect_sizes:
                n = mcnemars_test_sample_size(
                    torch.tensor(es),
                    torch.tensor(current_discord),
                    power=self.power,
                    alpha=self.alpha,
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
                label=f"Current (ES={current_effect:.3f}, N={current_n:.0f})",
            )

            # Add reference lines
            ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="N = 100")
            ax.axhline(y=200, color="gray", linestyle=":", alpha=0.5, label="N = 200")

            ax.set_xlabel("Effect Size (Discordant Difference)")
            ax.set_ylabel("Required Sample Size")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = (
                    f"McNemar's Test Sample Size (Power={self.power}, α={self.alpha})"
                )
        else:
            raise ValueError(f"plot_type must be 'sample_size_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha})"
