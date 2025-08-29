from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import mixed_model_sample_size


class MixedModelSampleSize(Metric):
    r"""
    Compute required sample size for mixed-effects models.

    This metric calculates the required sample size for mixed-effects models
    to achieve a desired statistical power. It follows TorchMetrics conventions
    for stateful metric computation.

    Parameters
    ----------
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import MixedModelSampleSize
    >>> metric = MixedModelSampleSize()
    >>> effect_size = torch.tensor(0.5)
    >>> cluster_size = torch.tensor(10)
    >>> intraclass_correlation = torch.tensor(0.1)
    >>> metric.update(effect_size, cluster_size, intraclass_correlation)
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
        self.add_state("cluster_size_values", default=[], dist_reduce_fx="cat")
        self.add_state(
            "intraclass_correlation_values",
            default=[],
            dist_reduce_fx="cat",
        )

    def update(
        self,
        effect_size: Tensor,
        cluster_size: Tensor,
        intraclass_correlation: Tensor,
    ) -> None:
        """
        Update the metric state with mixed model parameters.

        Parameters
        ----------
        effect_size : Tensor
            Effect size of the fixed effect.
        cluster_size : Tensor
            Average cluster size.
        intraclass_correlation : Tensor
            Intraclass correlation coefficient.
        """
        self.effect_size_values.append(effect_size)
        self.cluster_size_values.append(cluster_size)
        self.intraclass_correlation_values.append(intraclass_correlation)

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
            or not self.cluster_size_values
            or not self.intraclass_correlation_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        effect_size = self.effect_size_values[-1]
        cluster_size = self.cluster_size_values[-1]
        icc = self.intraclass_correlation_values[-1]

        # Use functional implementation
        return mixed_model_sample_size(
            effect_size,
            cluster_size,
            icc,
            power=self.power,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.effect_size_values = []
        self.cluster_size_values = []
        self.intraclass_correlation_values = []

    def plot(
        self,
        plot_type: str = "sample_size_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot mixed model sample size visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if (
            not self.effect_size_values
            or not self.cluster_size_values
            or not self.intraclass_correlation_values
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
            current_cluster = float(self.cluster_size_values[-1])
            current_icc = float(self.intraclass_correlation_values[-1])

            # Create range of effect sizes
            effect_sizes = np.linspace(0.1, 1.0, 100)
            sample_sizes = []

            for es in effect_sizes:
                n = mixed_model_sample_size(
                    torch.tensor(es),
                    torch.tensor(current_cluster),
                    torch.tensor(current_icc),
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
                label=f"Current (ES={current_effect:.2f}, N={current_n:.0f})",
            )

            # Add reference lines
            ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="N = 100")
            ax.axhline(y=200, color="gray", linestyle=":", alpha=0.5, label="N = 200")

            ax.set_xlabel("Effect Size")
            ax.set_ylabel("Required Sample Size")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"Mixed Model Sample Size (Power={self.power}, α={self.alpha}, ICC={current_icc:.2f})"
        else:
            raise ValueError(f"plot_type must be 'sample_size_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha})"
