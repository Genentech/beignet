from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import cohens_kappa_power


class CohensKappaPower(Metric):
    r"""
    Compute statistical power for Cohen's Kappa test.

    This metric calculates the statistical power for Cohen's Kappa coefficient
    test, which measures inter-rater agreement. It follows TorchMetrics
    conventions for stateful metric computation.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import CohensKappaPower
    >>> metric = CohensKappaPower()
    >>> kappa0 = torch.tensor(0.0)  # Null hypothesis kappa
    >>> kappa1 = torch.tensor(0.5)  # Alternative hypothesis kappa
    >>> sample_size = torch.tensor(100)
    >>> metric.update(kappa0, kappa1, sample_size)
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
        self.add_state("kappa0_values", default=[], dist_reduce_fx="cat")
        self.add_state("kappa1_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        kappa0: Tensor,
        kappa1: Tensor,
        sample_size: Tensor,
    ) -> None:
        """
        Update the metric state with power analysis parameters.

        Parameters
        ----------
        kappa0 : Tensor
            Null hypothesis kappa value.
        kappa1 : Tensor
            Alternative hypothesis kappa value.
        sample_size : Tensor
            Sample size.
        """
        self.kappa0_values.append(kappa0)
        self.kappa1_values.append(kappa1)
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
            not self.kappa0_values
            or not self.kappa1_values
            or not self.sample_size_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        kappa0 = self.kappa0_values[-1]
        kappa1 = self.kappa1_values[-1]
        sample_size = self.sample_size_values[-1]

        # Use functional implementation
        return cohens_kappa_power(
            kappa0,
            kappa1,
            sample_size,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.kappa0_values = []
        self.kappa1_values = []
        self.sample_size_values = []

    def plot(
        self,
        plot_type: str = "power_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Cohen's Kappa power visualization.

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
            not self.kappa0_values
            or not self.kappa1_values
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
            # Get current parameters
            current_kappa0 = float(self.kappa0_values[-1])
            current_kappa1 = float(self.kappa1_values[-1])
            current_n = float(self.sample_size_values[-1])

            # Create range of alternative kappa values
            kappa1_values = np.linspace(current_kappa0 + 0.1, 1.0, 100)
            powers = []

            for k1 in kappa1_values:
                power = cohens_kappa_power(
                    torch.tensor(current_kappa0),
                    torch.tensor(k1),
                    torch.tensor(current_n),
                    alpha=self.alpha,
                )
                powers.append(float(power))

            ax.plot(kappa1_values, powers, **kwargs)

            # Mark current point
            current_power = float(self.compute())
            ax.plot(
                current_kappa1,
                current_power,
                "ro",
                markersize=8,
                label=f"Current (κ₁={current_kappa1:.2f}, Power={current_power:.3f})",
            )

            # Add reference lines
            ax.axhline(
                y=0.8, color="gray", linestyle="--", alpha=0.5, label="Power = 0.8"
            )
            ax.axhline(
                y=0.5, color="gray", linestyle=":", alpha=0.5, label="Power = 0.5"
            )

            ax.set_xlabel("Alternative Kappa (κ₁)")
            ax.set_ylabel("Statistical Power")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"Cohen's Kappa Power Analysis (κ₀={current_kappa0}, N={current_n}, α={self.alpha})"
        else:
            raise ValueError(f"plot_type must be 'power_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"
