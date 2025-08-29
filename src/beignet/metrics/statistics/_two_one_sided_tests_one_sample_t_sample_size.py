from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import two_one_sided_tests_one_sample_t_sample_size


class TwoOneSidedTestsOneSampleTSampleSize(Metric):
    r"""
    Compute required sample size for two one-sided tests (TOST) one-sample t-test.

    This metric calculates the required sample size for TOST one-sample t-test
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
    >>> from beignet.metrics import TwoOneSidedTestsOneSampleTSampleSize
    >>> metric = TwoOneSidedTestsOneSampleTSampleSize()
    >>> effect_size = torch.tensor(0.1)
    >>> equivalence_margin = torch.tensor(0.5)
    >>> metric.update(effect_size, equivalence_margin)
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
        self.add_state("margin_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        effect_size: Tensor,
        equivalence_margin: Tensor,
    ) -> None:
        """
        Update the metric state with TOST parameters.

        Parameters
        ----------
        effect_size : Tensor
            True effect size.
        equivalence_margin : Tensor
            Equivalence margin (creates symmetric bounds [-margin, +margin]).
        """
        self.effect_size_values.append(torch.atleast_1d(effect_size))
        self.margin_values.append(torch.atleast_1d(equivalence_margin))

    def compute(self) -> Tensor:
        """
        Compute required sample size based on stored parameters.

        Returns
        -------
        Tensor
            The computed required sample size.
        """
        if not self.effect_size_values or not self.margin_values:
            raise RuntimeError("No values have been added to the metric.")

        effect_size_tensor = torch.cat(self.effect_size_values, dim=0)
        margin_tensor = torch.cat(self.margin_values, dim=0)

        # Create symmetric equivalence bounds: [-margin, +margin]
        lower_bound = -margin_tensor
        upper_bound = margin_tensor
        true_effect = effect_size_tensor

        return two_one_sided_tests_one_sample_t_sample_size(
            lower_bound,
            upper_bound,
            true_effect,
            power=self.power,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.effect_size_values = []
        self.margin_values = []

    def plot(
        self,
        plot_type: str = "sample_size_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot TOST one-sample t-test sample size visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if not self.effect_size_values or not self.margin_values:
            raise RuntimeError(
                "No values have been added to the metric. Call update() first.",
            )

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        if plot_type == "sample_size_curve":
            current_margin = float(self.margin_values[-1])
            current_lower = -current_margin
            current_upper = current_margin
            current_effect = float(self.effect_size_values[-1])

            # Create range of true effects within equivalence bounds
            min_effect = min(current_lower, current_upper)
            max_effect = max(current_lower, current_upper)
            true_effects = np.linspace(min_effect * 0.8, max_effect * 0.8, 100)
            sample_sizes = []

            for effect in true_effects:
                n = two_one_sided_tests_one_sample_t_sample_size(
                    torch.tensor(current_lower),
                    torch.tensor(current_upper),
                    torch.tensor(effect),
                    power=self.power,
                    alpha=self.alpha,
                )
                sample_sizes.append(float(n))

            ax.plot(true_effects, sample_sizes, **kwargs)

            # Mark current point
            current_n = float(self.compute())
            ax.plot(
                current_effect,
                current_n,
                "ro",
                markersize=8,
                label=f"Current (Effect={current_effect:.2f}, N={current_n:.0f})",
            )

            # Mark equivalence bounds
            ax.axvline(
                x=current_lower,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"Lower Bound = {current_lower:.2f}",
            )
            ax.axvline(
                x=current_upper,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"Upper Bound = {current_upper:.2f}",
            )

            ax.set_xlabel("True Effect Size")
            ax.set_ylabel("Required Sample Size")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"TOST One-Sample T-Test Sample Size (Power={self.power}, α={self.alpha})"
        else:
            raise ValueError(f"plot_type must be 'sample_size_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha})"
