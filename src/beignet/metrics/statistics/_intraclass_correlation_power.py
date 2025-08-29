from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import intraclass_correlation_power


class IntraclassCorrelationPower(Metric):
    r"""
    Compute statistical power for intraclass correlation analysis.

    This metric calculates the statistical power for intraclass correlation
    coefficient (ICC) analysis. It follows TorchMetrics conventions for
    stateful metric computation.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import IntraclassCorrelationPower
    >>> metric = IntraclassCorrelationPower()
    >>> icc_value = torch.tensor(0.7)
    >>> sample_size = torch.tensor(30)
    >>> raters = torch.tensor(3)
    >>> metric.update(icc_value, sample_size, raters)
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
        self.add_state("icc_values", default=[], dist_reduce_fx="cat")
        self.add_state("sample_size_values", default=[], dist_reduce_fx="cat")
        self.add_state("raters_values", default=[], dist_reduce_fx="cat")

    def update(self, icc_value: Tensor, sample_size: Tensor, raters: Tensor) -> None:
        """
        Update the metric state with ICC parameters.

        Parameters
        ----------
        icc_value : Tensor
            Expected intraclass correlation coefficient.
        sample_size : Tensor
            Number of subjects.
        raters : Tensor
            Number of raters.
        """
        self.icc_values.append(icc_value)
        self.sample_size_values.append(sample_size)
        self.raters_values.append(raters)

    def compute(self) -> Tensor:
        """
        Compute statistical power based on stored parameters.

        Returns
        -------
        Tensor
            The computed statistical power.
        """
        if not self.icc_values or not self.sample_size_values or not self.raters_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        icc_value = self.icc_values[-1]
        sample_size = self.sample_size_values[-1]
        raters = self.raters_values[-1]

        # Use functional implementation
        return intraclass_correlation_power(
            icc_value,
            sample_size,
            raters,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.icc_values = []
        self.sample_size_values = []
        self.raters_values = []

    def plot(
        self,
        plot_type: str = "power_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot intraclass correlation power visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.icc_values or not self.sample_size_values or not self.raters_values:
            raise RuntimeError(
                "No values have been added to the metric. Call update() first.",
            )

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "power_curve":
            current_icc = float(self.icc_values[-1])
            current_n = float(self.sample_size_values[-1])
            current_raters = float(self.raters_values[-1])

            # Create range of ICC values
            icc_values = np.linspace(0.1, 0.95, 100)
            powers = []

            for icc in icc_values:
                power = intraclass_correlation_power(
                    torch.tensor(icc),
                    torch.tensor(current_n),
                    torch.tensor(current_raters),
                    alpha=self.alpha,
                )
                powers.append(float(power))

            ax.plot(icc_values, powers, **kwargs)

            # Mark current point
            current_power = float(self.compute())
            ax.plot(
                current_icc,
                current_power,
                "ro",
                markersize=8,
                label=f"Current (ICC={current_icc:.2f}, Power={current_power:.3f})",
            )

            # Add reference lines
            ax.axhline(
                y=0.8, color="gray", linestyle="--", alpha=0.5, label="Power = 0.8"
            )
            ax.axhline(
                y=0.5, color="gray", linestyle=":", alpha=0.5, label="Power = 0.5"
            )

            ax.set_xlabel("Intraclass Correlation Coefficient")
            ax.set_ylabel("Statistical Power")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"ICC Power Analysis (α={self.alpha}, N={current_n}, Raters={current_raters})"
        else:
            raise ValueError(f"plot_type must be 'power_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"
