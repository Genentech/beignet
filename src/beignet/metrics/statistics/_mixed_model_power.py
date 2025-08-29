from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import mixed_model_power


class MixedModelPower(Metric):
    r"""
    Compute statistical power for mixed-effects models.

    This metric calculates the statistical power for mixed-effects models,
    which account for both fixed and random effects. It follows TorchMetrics
    conventions for stateful metric computation.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import MixedModelPower
    >>> metric = MixedModelPower()
    >>> effect_size = torch.tensor(0.5)
    >>> sample_size = torch.tensor(100)
    >>> cluster_size = torch.tensor(10)
    >>> intraclass_correlation = torch.tensor(0.1)
    >>> metric.update(effect_size, sample_size, cluster_size, intraclass_correlation)
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
        self.add_state("cluster_size_values", default=[], dist_reduce_fx="cat")
        self.add_state(
            "intraclass_correlation_values", default=[], dist_reduce_fx="cat"
        )

    def update(
        self,
        effect_size: Tensor,
        sample_size: Tensor,
        cluster_size: Tensor,
        intraclass_correlation: Tensor,
    ) -> None:
        """
        Update the metric state with mixed model parameters.

        Parameters
        ----------
        effect_size : Tensor
            Effect size of the fixed effect.
        sample_size : Tensor
            Total sample size.
        cluster_size : Tensor
            Average cluster size.
        intraclass_correlation : Tensor
            Intraclass correlation coefficient.
        """
        self.effect_size_values.append(effect_size)
        self.sample_size_values.append(sample_size)
        self.cluster_size_values.append(cluster_size)
        self.intraclass_correlation_values.append(intraclass_correlation)

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
            or not self.cluster_size_values
            or not self.intraclass_correlation_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        effect_size = self.effect_size_values[-1]
        sample_size = self.sample_size_values[-1]
        cluster_size = self.cluster_size_values[-1]
        icc = self.intraclass_correlation_values[-1]

        # Use functional implementation
        return mixed_model_power(
            effect_size,
            sample_size,
            cluster_size,
            icc,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.effect_size_values = []
        self.sample_size_values = []
        self.cluster_size_values = []
        self.intraclass_correlation_values = []

    def plot(
        self,
        plot_type: str = "power_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot mixed model power visualization."""
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

        if plot_type == "power_curve":
            current_effect = float(self.effect_size_values[-1])
            current_n = float(self.sample_size_values[-1])
            current_cluster = float(self.cluster_size_values[-1])
            current_icc = float(self.intraclass_correlation_values[-1])

            # Create range of effect sizes
            effect_sizes = np.linspace(0.1, 1.0, 100)
            powers = []

            for es in effect_sizes:
                power = mixed_model_power(
                    torch.tensor(es),
                    torch.tensor(current_n),
                    torch.tensor(current_cluster),
                    torch.tensor(current_icc),
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
                title = f"Mixed Model Power (α={self.alpha}, N={current_n}, ICC={current_icc:.2f})"
        else:
            raise ValueError(f"plot_type must be 'power_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"
