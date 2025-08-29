from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import mcnemars_test_power


class McnemarsTestPower(Metric):
    r"""
    Compute statistical power for McNemar's test.

    This metric calculates the statistical power for McNemar's test, which tests
    for differences in paired binary data. It follows TorchMetrics conventions
    for stateful metric computation.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import McnemarsTestPower
    >>> metric = McnemarsTestPower()
    >>> prob_discordant = torch.tensor(0.2)
    >>> odds_ratio = torch.tensor(2.0)
    >>> sample_size = torch.tensor(100)
    >>> metric.update(prob_discordant, odds_ratio, sample_size)
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
        self.add_state("prob_discordant_values", default=[], dist_reduce_fx="cat")
        self.add_state("odds_ratio_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        prob_discordant: Tensor,
        odds_ratio: Tensor,
        sample_size: Tensor,
    ) -> None:
        """
        Update the metric state with test parameters.

        Parameters
        ----------
        prob_discordant : Tensor
            Probability of discordant pairs.
        odds_ratio : Tensor
            Expected odds ratio.
        sample_size : Tensor
            Sample size.
        """
        self.prob_discordant_values.append(prob_discordant)
        self.odds_ratio_values.append(odds_ratio)
        self.sample_size_values.append(sample_size)

    def compute(self) -> Tensor:
        """
        Compute statistical power based on stored parameters.

        Returns
        -------
        Tensor
            The computed statistical power.
        """
        if (
            not self.prob_discordant_values
            or not self.odds_ratio_values
            or not self.sample_size_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        prob_discordant = self.prob_discordant_values[-1]
        odds_ratio = self.odds_ratio_values[-1]
        sample_size = self.sample_size_values[-1]

        # Use functional implementation
        return mcnemars_test_power(
            prob_discordant,
            odds_ratio,
            sample_size,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.prob_discordant_values = []
        self.odds_ratio_values = []
        self.sample_size_values = []

    def plot(
        self,
        plot_type: str = "power_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot McNemar's test power visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if (
            not self.prob_discordant_values
            or not self.odds_ratio_values
            or not self.sample_size_values
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
            current_pd = float(self.prob_discordant_values[-1])
            current_or = float(self.odds_ratio_values[-1])
            current_n = float(self.sample_size_values[-1])

            # Create range of odds ratios
            odds_ratios = np.linspace(1.2, 4.0, 100)
            powers = []

            for or_val in odds_ratios:
                power = mcnemars_test_power(
                    torch.tensor(current_pd),
                    torch.tensor(or_val),
                    torch.tensor(current_n),
                    alpha=self.alpha,
                )
                powers.append(float(power))

            ax.plot(odds_ratios, powers, **kwargs)

            # Mark current point
            current_power = float(self.compute())
            ax.plot(
                current_or,
                current_power,
                "ro",
                markersize=8,
                label=f"Current (OR={current_or:.2f}, Power={current_power:.3f})",
            )

            ax.axhline(
                y=0.8,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Power = 0.8",
            )
            ax.set_xlabel("Odds Ratio")
            ax.set_ylabel("Statistical Power")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"McNemar's Test Power Analysis (α={self.alpha}, N={current_n})"
        else:
            raise ValueError(f"plot_type must be 'power_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"
