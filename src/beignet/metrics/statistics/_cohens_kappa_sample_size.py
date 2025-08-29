from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import cohens_kappa_sample_size


class CohensKappaSampleSize(Metric):
    r"""
    Compute required sample size for Cohen's Kappa test.

    This metric calculates the required sample size for Cohen's Kappa coefficient
    test to achieve a desired statistical power. It follows TorchMetrics
    conventions for stateful metric computation.

    Parameters
    ----------
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import CohensKappaSampleSize
    >>> metric = CohensKappaSampleSize()
    >>> kappa0 = torch.tensor(0.0)  # Null hypothesis kappa
    >>> kappa1 = torch.tensor(0.5)  # Alternative hypothesis kappa
    >>> metric.update(kappa0, kappa1)
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
        self.add_state("kappa0_values", default=[], dist_reduce_fx="cat")
        self.add_state("kappa1_values", default=[], dist_reduce_fx="cat")

    def update(self, kappa0: Tensor, kappa1: Tensor) -> None:
        """
        Update the metric state with kappa values.

        Parameters
        ----------
        kappa0 : Tensor
            Null hypothesis kappa value.
        kappa1 : Tensor
            Alternative hypothesis kappa value.
        """
        self.kappa0_values.append(kappa0)
        self.kappa1_values.append(kappa1)

    def compute(self) -> Tensor:
        """
        Compute required sample size based on stored parameters.

        Returns
        -------
        Tensor
            The computed required sample size.
        """
        if not self.kappa0_values or not self.kappa1_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        kappa0 = self.kappa0_values[-1]
        kappa1 = self.kappa1_values[-1]

        # Use functional implementation
        return cohens_kappa_sample_size(
            kappa0,
            kappa1,
            power=self.power,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.kappa0_values = []
        self.kappa1_values = []

    def plot(
        self,
        plot_type: str = "sample_size_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Cohen's Kappa sample size visualization.

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

        if not self.kappa0_values or not self.kappa1_values:
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
            current_kappa0 = float(self.kappa0_values[-1])
            current_kappa1 = float(self.kappa1_values[-1])

            # Create range of alternative kappa values
            kappa1_values = np.linspace(current_kappa0 + 0.1, 1.0, 100)
            sample_sizes = []

            for k1 in kappa1_values:
                n = cohens_kappa_sample_size(
                    torch.tensor(current_kappa0),
                    torch.tensor(k1),
                    power=self.power,
                    alpha=self.alpha,
                )
                sample_sizes.append(float(n))

            ax.plot(kappa1_values, sample_sizes, **kwargs)

            # Mark current point
            current_n = float(self.compute())
            ax.plot(
                current_kappa1,
                current_n,
                "ro",
                markersize=8,
                label=f"Current (κ₁={current_kappa1:.2f}, N={current_n:.0f})",
            )

            # Add reference lines for common sample sizes
            ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="N = 50")
            ax.axhline(y=100, color="gray", linestyle=":", alpha=0.5, label="N = 100")

            ax.set_xlabel("Alternative Kappa (κ₁)")
            ax.set_ylabel("Required Sample Size")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"Cohen's Kappa Sample Size (κ₀={current_kappa0}, Power={self.power}, α={self.alpha})"
        else:
            raise ValueError(f"plot_type must be 'sample_size_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha})"
