from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import intraclass_correlation_sample_size


class IntraclassCorrelationSampleSize(Metric):
    r"""
    Compute required sample size for intraclass correlation analysis.

    This metric calculates the required sample size for intraclass correlation
    coefficient (ICC) analysis to achieve a desired statistical power. It follows
    TorchMetrics conventions for stateful metric computation.

    Parameters
    ----------
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import IntraclassCorrelationSampleSize
    >>> metric = IntraclassCorrelationSampleSize()
    >>> icc_value = torch.tensor(0.7)
    >>> raters = torch.tensor(3)
    >>> metric.update(icc_value, raters)
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
        self.add_state("icc_values", default=[], dist_reduce_fx="cat")
        self.add_state("raters_values", default=[], dist_reduce_fx="cat")

    def update(self, icc_value: Tensor, raters: Tensor) -> None:
        """
        Update the metric state with ICC parameters.

        Parameters
        ----------
        icc_value : Tensor
            Expected intraclass correlation coefficient.
        raters : Tensor
            Number of raters.
        """
        self.icc_values.append(icc_value)
        self.raters_values.append(raters)

    def compute(self) -> Tensor:
        """
        Compute required sample size based on stored parameters.

        Returns
        -------
        Tensor
            The computed required sample size (number of subjects).
        """
        if not self.icc_values or not self.raters_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        icc_value = self.icc_values[-1]
        raters = self.raters_values[-1]

        # Use functional implementation
        return intraclass_correlation_sample_size(
            icc_value,
            raters,
            power=self.power,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.icc_values = []
        self.raters_values = []

    def plot(
        self,
        plot_type: str = "sample_size_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot intraclass correlation sample size visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.icc_values or not self.raters_values:
            raise RuntimeError(
                "No values have been added to the metric. Call update() first.",
            )

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "sample_size_curve":
            current_icc = float(self.icc_values[-1])
            current_raters = float(self.raters_values[-1])

            # Create range of ICC values
            icc_values = np.linspace(0.1, 0.95, 100)
            sample_sizes = []

            for icc in icc_values:
                n = intraclass_correlation_sample_size(
                    torch.tensor(icc),
                    torch.tensor(current_raters),
                    power=self.power,
                    alpha=self.alpha,
                )
                sample_sizes.append(float(n))

            ax.plot(icc_values, sample_sizes, **kwargs)

            # Mark current point
            current_n = float(self.compute())
            ax.plot(
                current_icc,
                current_n,
                "ro",
                markersize=8,
                label=f"Current (ICC={current_icc:.2f}, N={current_n:.0f})",
            )

            # Add reference lines for sample sizes
            ax.axhline(y=20, color="gray", linestyle="--", alpha=0.5, label="N = 20")
            ax.axhline(y=50, color="gray", linestyle=":", alpha=0.5, label="N = 50")

            ax.set_xlabel("Intraclass Correlation Coefficient")
            ax.set_ylabel("Required Sample Size (Subjects)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"ICC Sample Size (Power={self.power}, α={self.alpha}, Raters={current_raters})"
        else:
            raise ValueError(f"plot_type must be 'sample_size_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha})"
