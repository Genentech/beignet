from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import friedman_test_power


class FriedmanTestPower(Metric):
    r"""
    Compute statistical power for Friedman test.

    This metric calculates the statistical power for Friedman test, which is
    a non-parametric test for repeated measures with multiple groups.
    It follows TorchMetrics conventions for stateful metric computation.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import FriedmanTestPower
    >>> metric = FriedmanTestPower()
    >>> effect_size = torch.tensor(0.3)
    >>> sample_size = torch.tensor(20)
    >>> groups = torch.tensor(4)
    >>> metric.update(effect_size, sample_size, groups)
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
        self.add_state("effect_size_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size_values", default=[], dist_reduce_fx="cat")
        self.add_state("groups_values", default=[], dist_reduce_fx="cat")

    def update(self, effect_size: Tensor, sample_size: Tensor, groups: Tensor) -> None:
        """
        Update the metric state with test parameters.

        Parameters
        ----------
        effect_size : Tensor
            Effect size measure.
        sample_size : Tensor
            Sample size (number of subjects).
        groups : Tensor
            Number of groups/conditions.
        """
        self.effect_size_values.append(effect_size)
        self.sample_size_values.append(sample_size)
        self.groups_values.append(groups)

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
            or not self.sample_size_values
            or not self.groups_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        effect_size = self.effect_size_values[-1]
        sample_size = self.sample_size_values[-1]
        groups = self.groups_values[-1]

        # Use functional implementation
        return friedman_test_power(
            effect_size,
            sample_size,
            groups,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.effect_size_values = []
        self.sample_size_values = []
        self.groups_values = []

    def plot(
        self,
        plot_type: str = "power_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Friedman test power visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if (
            not self.effect_size_values
            or not self.sample_size_values
            or not self.groups_values
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
            current_n = float(self.sample_size_values[-1])
            current_groups = float(self.groups_values[-1])

            # Create range of effect sizes
            effect_sizes = np.linspace(0.1, 1.0, 100)
            powers = []

            for es in effect_sizes:
                power = friedman_test_power(
                    torch.tensor(es),
                    torch.tensor(current_n),
                    torch.tensor(current_groups),
                    alpha=self.alpha,
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
                title = f"Friedman Test Power Analysis (α={self.alpha}, N={current_n}, Groups={current_groups})"
        else:
            raise ValueError(f"plot_type must be 'power_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"
