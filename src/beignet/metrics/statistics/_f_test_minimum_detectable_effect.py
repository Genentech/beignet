from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import f_test_minimum_detectable_effect


class FTestMinimumDetectableEffect(Metric):
    r"""
    Compute minimum detectable effect for F-test.

    This metric calculates the minimum detectable effect size for F-test
    given sample size and power. It follows TorchMetrics conventions for
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
    >>> from beignet.metrics import FTestMinimumDetectableEffect
    >>> metric = FTestMinimumDetectableEffect()
    >>> sample_size = torch.tensor(60)
    >>> degrees_of_freedom_1 = torch.tensor(2)
    >>> degrees_of_freedom_2 = torch.tensor(57)
    >>> metric.update(sample_size, degrees_of_freedom_1, degrees_of_freedom_2)
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
        self.add_state("degrees_of_freedom_1_values", default=[], dist_reduce_fx="cat")
        self.add_state("degrees_of_freedom_2_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        sample_size: Tensor,
        degrees_of_freedom_1: Tensor,
        degrees_of_freedom_2: Tensor,
    ) -> None:
        """
        Update the metric state with F-test parameters.

        Parameters
        ----------
        sample_size : Tensor
            Sample size.
        degrees_of_freedom_1 : Tensor
            Numerator degrees of freedom.
        degrees_of_freedom_2 : Tensor
            Denominator degrees of freedom.
        """
        self.sample_size_values.append(sample_size)
        self.degrees_of_freedom_1_values.append(degrees_of_freedom_1)
        self.degrees_of_freedom_2_values.append(degrees_of_freedom_2)

    def compute(self) -> Tensor:
        """
        Compute minimum detectable effect based on stored parameters.

        Returns
        -------
        Tensor
            The computed minimum detectable effect size.
        """
        if (
            not self.sample_size_values
            or not self.degrees_of_freedom_1_values
            or not self.degrees_of_freedom_2_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        sample_size = self.sample_size_values[-1]
        df1 = self.degrees_of_freedom_1_values[-1]
        df2 = self.degrees_of_freedom_2_values[-1]

        # Use functional implementation
        return f_test_minimum_detectable_effect(
            sample_size,
            df1,
            df2,
            power=self.power,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.sample_size_values = []
        self.degrees_of_freedom_1_values = []
        self.degrees_of_freedom_2_values = []

    def plot(
        self,
        plot_type: str = "effect_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot F-test minimum detectable effect visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if (
            not self.sample_size_values
            or not self.degrees_of_freedom_1_values
            or not self.degrees_of_freedom_2_values
        ):
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
            current_df1 = float(self.degrees_of_freedom_1_values[-1])
            current_df2 = float(self.degrees_of_freedom_2_values[-1])

            # Create range of sample sizes
            sample_sizes = np.arange(20, 200, 5)
            effects = []

            for n in sample_sizes:
                # Adjust df2 proportionally with sample size
                df2_adjusted = current_df2 * (n / current_n)
                effect = f_test_minimum_detectable_effect(
                    torch.tensor(n),
                    torch.tensor(current_df1),
                    torch.tensor(df2_adjusted),
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
                title = f"F-Test MDE (Power={self.power}, α={self.alpha})"
        else:
            raise ValueError(f"plot_type must be 'effect_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha})"
