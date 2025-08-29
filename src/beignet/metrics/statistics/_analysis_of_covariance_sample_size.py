from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import analysis_of_covariance_sample_size


class AnalysisOfCovarianceSampleSize(Metric):
    r"""
    Compute required sample size for Analysis of Covariance (ANCOVA).

    This metric calculates the required sample size for ANCOVA to achieve
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
    >>> from beignet.metrics import AnalysisOfCovarianceSampleSize
    >>> metric = AnalysisOfCovarianceSampleSize()
    >>> effect_size = torch.tensor(0.25)
    >>> groups = torch.tensor(3)
    >>> covariate_r2 = torch.tensor(0.2)
    >>> num_covariates = torch.tensor(1)
    >>> metric.update(effect_size, groups, covariate_r2, num_covariates)
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
        self.add_state("effect_size_values", default=[], dist_reduce_fx="cat")
        self.add_state("groups_values", default=[], dist_reduce_fx="cat")
        self.add_state("covariate_r2_values", default=[], dist_reduce_fx="cat")
        self.add_state("num_covariates_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        effect_size: Tensor,
        groups: Tensor,
        covariate_r2: Tensor,
        num_covariates: Tensor,
    ) -> None:
        """
        Update the metric state with ANCOVA parameters.

        Parameters
        ----------
        effect_size : Tensor
            Effect size (f).
        groups : Tensor
            Number of groups.
        covariate_r2 : Tensor
            R-squared of covariates with dependent variable.
        num_covariates : Tensor
            Number of covariates.
        """
        self.effect_size_values.append(effect_size)
        self.groups_values.append(groups)
        self.covariate_r2_values.append(covariate_r2)
        self.num_covariates_values.append(num_covariates)

    def compute(self) -> Tensor:
        """
        Compute required sample size based on stored parameters.

        Returns
        -------
        Tensor
            The computed required sample size.
        """
        if (
            not self.effect_size_values
            or not self.groups_values
            or not self.covariate_r2_values
            or not self.num_covariates_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        effect_size = self.effect_size_values[-1]
        groups = self.groups_values[-1]
        covariate_r2 = self.covariate_r2_values[-1]
        num_covariates = self.num_covariates_values[-1]

        # Use functional implementation
        return analysis_of_covariance_sample_size(
            effect_size,
            groups,
            covariate_r2,
            num_covariates,
            power=self.power,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.effect_size_values = []
        self.groups_values = []
        self.covariate_r2_values = []
        self.num_covariates_values = []

    def plot(
        self,
        plot_type: str = "sample_size_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot ANCOVA sample size visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if (
            not self.effect_size_values
            or not self.groups_values
            or not self.covariate_r2_values
            or not self.num_covariates_values
        ):
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
            current_groups = float(self.groups_values[-1])
            current_r2 = float(self.covariate_r2_values[-1])
            current_covs = float(self.num_covariates_values[-1])

            # Create range of effect sizes
            effect_sizes = np.linspace(0.1, 1.0, 100)
            sample_sizes = []

            for es in effect_sizes:
                n = analysis_of_covariance_sample_size(
                    torch.tensor(es),
                    torch.tensor(current_groups),
                    torch.tensor(current_r2),
                    torch.tensor(current_covs),
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
                label=f"Current (f={current_effect:.2f}, N={current_n:.0f})",
            )

            # Add reference lines
            ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="N = 50")
            ax.axhline(y=100, color="gray", linestyle=":", alpha=0.5, label="N = 100")

            ax.set_xlabel("Effect Size (f)")
            ax.set_ylabel("Required Sample Size")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"ANCOVA Sample Size (Power={self.power}, α={self.alpha})"
        else:
            raise ValueError(f"plot_type must be 'sample_size_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha})"
