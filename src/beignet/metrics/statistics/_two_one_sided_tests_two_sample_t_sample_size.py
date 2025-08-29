from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from ..functional.statistics import two_one_sided_tests_two_sample_t_sample_size


class TwoOneSidedTestsTwoSampleTSampleSize(Metric):
    r"""
    Compute required sample size for two one-sided tests (TOST) two-sample t-test.

    This metric calculates the required sample size for TOST two-sample t-test
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
    >>> from beignet.metrics import TwoOneSidedTestsTwoSampleTSampleSize
    >>> metric = TwoOneSidedTestsTwoSampleTSampleSize()
    >>> lower_equivalence_bound = torch.tensor(-0.5)
    >>> upper_equivalence_bound = torch.tensor(0.5)
    >>> true_difference = torch.tensor(0.1)
    >>> metric.update(lower_equivalence_bound, upper_equivalence_bound, true_difference)
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
        self.add_state("lower_bound_values", default=[], dist_reduce_fx="cat")
        self.add_state("upper_bound_values", default=[], dist_reduce_fx="cat")
        self.add_state("true_difference_values", default=[], dist_reduce_fx="cat")

    def update(
        self,
        lower_equivalence_bound: Tensor,
        upper_equivalence_bound: Tensor,
        true_difference: Tensor,
    ) -> None:
        """
        Update the metric state with TOST parameters.

        Parameters
        ----------
        lower_equivalence_bound : Tensor
            Lower bound for equivalence region.
        upper_equivalence_bound : Tensor
            Upper bound for equivalence region.
        true_difference : Tensor
            True difference between groups.
        """
        self.lower_bound_values.append(lower_equivalence_bound)
        self.upper_bound_values.append(upper_equivalence_bound)
        self.true_difference_values.append(true_difference)

    def compute(self) -> Tensor:
        """
        Compute required sample size based on stored parameters.

        Returns
        -------
        Tensor
            The computed required sample size.
        """
        if (
            not self.lower_bound_values
            or not self.upper_bound_values
            or not self.true_difference_values
        ):
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values
        lower_bound = self.lower_bound_values[-1]
        upper_bound = self.upper_bound_values[-1]
        true_difference = self.true_difference_values[-1]

        # Use functional implementation
        return two_one_sided_tests_two_sample_t_sample_size(
            lower_bound,
            upper_bound,
            true_difference,
            power=self.power,
            alpha=self.alpha,
        )

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.lower_bound_values = []
        self.upper_bound_values = []
        self.true_difference_values = []

    def plot(
        self,
        plot_type: str = "sample_size_curve",
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot TOST two-sample t-test sample size visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib",
            ) from err

        if (
            not self.lower_bound_values
            or not self.upper_bound_values
            or not self.true_difference_values
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
            current_lower = float(self.lower_bound_values[-1])
            current_upper = float(self.upper_bound_values[-1])
            current_diff = float(self.true_difference_values[-1])

            # Create range of true differences within equivalence bounds
            min_diff = min(current_lower, current_upper)
            max_diff = max(current_lower, current_upper)
            true_differences = np.linspace(min_diff * 0.8, max_diff * 0.8, 100)
            sample_sizes = []

            for diff in true_differences:
                n = two_one_sided_tests_two_sample_t_sample_size(
                    torch.tensor(current_lower),
                    torch.tensor(current_upper),
                    torch.tensor(diff),
                    power=self.power,
                    alpha=self.alpha,
                )
                sample_sizes.append(float(n))

            ax.plot(true_differences, sample_sizes, **kwargs)

            # Mark current point
            current_n = float(self.compute())
            ax.plot(
                current_diff,
                current_n,
                "ro",
                markersize=8,
                label=f"Current (Diff={current_diff:.2f}, N={current_n:.0f})",
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

            ax.set_xlabel("True Difference")
            ax.set_ylabel("Required Sample Size")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if title is None:
                title = f"TOST Two-Sample T-Test Sample Size (Power={self.power}, α={self.alpha})"
        else:
            raise ValueError(f"plot_type must be 'sample_size_curve', got {plot_type}")

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(power={self.power}, alpha={self.alpha})"
