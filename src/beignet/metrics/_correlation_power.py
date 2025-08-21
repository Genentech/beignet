import math
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric


class CorrelationPower(Metric):
    r"""
    Compute statistical power for correlation tests based on expected correlation.

    This metric calculates the statistical power for detecting a correlation
    of a specified magnitude. It follows TorchMetrics conventions for
    stateful metric computation.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics import CorrelationPower
    >>> metric = CorrelationPower()
    >>> # Simulate data for expected correlation of 0.3 with n=50
    >>> n = 50
    >>> r_expected = 0.3
    >>> # You would update with your expected r and actual sample size
    >>> metric.update(torch.tensor(r_expected), torch.tensor(n))
    >>> metric.compute()
    tensor(...)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if alternative not in ["two-sided", "greater", "less"]:
            raise ValueError(f"Unknown alternative: {alternative}")
        if not 0 < alpha < 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

        self.alpha = alpha
        self.alternative = alternative

        # State for storing r and sample_size values
        self.add_state("r_values", default=[], dist_reduce_fx="cat")
        self.add_state("n_values", default=[], dist_reduce_fx="cat")

    def update(self, r: Tensor, sample_size: Tensor) -> None:
        """
        Update the metric state with expected correlation and sample size.

        Parameters
        ----------
        r : Tensor
            Expected correlation coefficient.
        sample_size : Tensor
            Sample size for the correlation test.
        """
        self.r_values.append(r)
        self.n_values.append(sample_size)

    def compute(self) -> Tensor:
        """
        Compute statistical power based on stored correlation and sample size values.

        Returns
        -------
        Tensor
            The computed statistical power (probability between 0 and 1).
        """
        if not self.r_values or not self.n_values:
            raise RuntimeError("No values have been added to the metric.")

        # Use the most recent values (or could average them)
        r = (
            self.r_values[-1]
            if len(self.r_values) == 1
            else torch.cat(self.r_values, dim=-1).mean()
        )
        n = (
            self.n_values[-1]
            if len(self.n_values) == 1
            else torch.cat(self.n_values, dim=-1).mean()
        )

        # Inline implementation of correlation_power to avoid circular imports

        # Convert inputs to tensors if needed
        r = torch.as_tensor(r)
        n = torch.as_tensor(n)

        # Ensure both have the same dtype
        if r.dtype != n.dtype:
            if r.dtype == torch.float64 or n.dtype == torch.float64:
                r = r.to(torch.float64)
                n = n.to(torch.float64)
            else:
                r = r.to(torch.float32)
                n = n.to(torch.float32)

        # Fisher z-transformation of the correlation
        # z_r = 0.5 * ln((1 + r) / (1 - r))
        epsilon = 1e-7  # Small value to avoid division by zero
        r_clamped = torch.clamp(r, -1 + epsilon, 1 - epsilon)
        z_r = 0.5 * torch.log((1 + r_clamped) / (1 - r_clamped))

        # Standard error of Fisher z-transform
        # SE = 1 / sqrt(n - 3)
        se_z = 1.0 / torch.sqrt(n - 3)

        # Test statistic under alternative hypothesis
        # z = z_r / SE
        z_stat = z_r / se_z

        # Critical values using standard normal approximation
        sqrt_2 = math.sqrt(2.0)

        if self.alternative == "two-sided":
            # Two-sided critical value
            z_alpha_2 = (
                torch.erfinv(torch.tensor(1 - self.alpha / 2, dtype=r.dtype)) * sqrt_2
            )

            # Power = P(|Z| > z_alpha/2 | H1) where Z ~ N(z_stat, 1)
            # This is 1 - P(-z_alpha/2 < Z < z_alpha/2 | H1)
            # = 1 - [Φ(z_alpha/2 - z_stat) - Φ(-z_alpha/2 - z_stat)]
            cdf_upper = 0.5 * (1 + torch.erf((z_alpha_2 - z_stat) / sqrt_2))
            cdf_lower = 0.5 * (1 + torch.erf((-z_alpha_2 - z_stat) / sqrt_2))
            power = 1 - (cdf_upper - cdf_lower)

        elif self.alternative == "greater":
            z_alpha = torch.erfinv(torch.tensor(1 - self.alpha, dtype=r.dtype)) * sqrt_2

            # Power = P(Z > z_alpha | H1) = 1 - Φ(z_alpha - z_stat)
            power = 1 - 0.5 * (1 + torch.erf((z_alpha - z_stat) / sqrt_2))

        elif self.alternative == "less":
            z_alpha = torch.erfinv(torch.tensor(self.alpha, dtype=r.dtype)) * sqrt_2

            # Power = P(Z < z_alpha | H1) = Φ(z_alpha - z_stat)
            power = 0.5 * (1 + torch.erf((z_alpha - z_stat) / sqrt_2))

        else:
            raise ValueError(f"Unknown alternative: {self.alternative}")

        # Clamp power to [0, 1] range
        return torch.clamp(power, 0.0, 1.0)

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.r_values = []
        self.n_values = []

    def plot(
        self,
        dep_var: str = "r",
        r: Optional[Any] = None,
        sample_size: Optional[Any] = None,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot correlation power curves varying different parameters.

        Parameters
        ----------
        dep_var : str, default="r"
            The variable to plot on x-axis. Either "r" or "sample_size".
        r : array-like, optional
            Range of correlation coefficients to plot. If None, uses reasonable default range.
        sample_size : array-like, optional
            Range of sample sizes to plot. If None, uses reasonable default range.
        alpha : float, default=0.05
            Significance level for power calculation.
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

        Examples
        --------
        >>> metric = CorrelationPower()
        >>> fig = metric.plot(dep_var="r", sample_size=[30, 50, 100])
        >>> fig = metric.plot(dep_var="sample_size", r=[0.1, 0.3, 0.5])
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            ) from err

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        # Set default ranges based on dep_var
        if dep_var == "r":
            if r is None:
                x_values = np.linspace(0.1, 0.8, 100)
            else:
                x_values = np.asarray(r)

            if sample_size is None:
                param_values = [30, 50, 100]  # Different sample sizes
                param_name = "Sample Size"
            else:
                param_values = np.asarray(sample_size)
                param_name = "Sample Size"

            x_label = "Correlation Coefficient (r)"

        elif dep_var == "sample_size":
            if sample_size is None:
                x_values = np.linspace(10, 200, 100)
            else:
                x_values = np.asarray(sample_size)

            if r is None:
                param_values = [0.1, 0.3, 0.5]  # Small, medium, large correlations
                param_name = "r"
            else:
                param_values = np.asarray(r)
                param_name = "r"

            x_label = "Sample Size"
        else:
            raise ValueError(f"dep_var must be 'r' or 'sample_size', got {dep_var}")

        # Plot power curves for different parameter values
        for param_val in param_values:
            y_values = []

            for x_val in x_values:
                if dep_var == "r":
                    r_val = torch.tensor(float(x_val))
                    n_val = torch.tensor(float(param_val))
                elif dep_var == "sample_size":
                    r_val = torch.tensor(float(param_val))
                    n_val = torch.tensor(float(x_val))

                # Inline power calculation
                # Fisher z-transformation of the correlation
                epsilon = 1e-7
                r_clamped = torch.clamp(r_val, -1 + epsilon, 1 - epsilon)
                z_r = 0.5 * torch.log((1 + r_clamped) / (1 - r_clamped))

                # Standard error of Fisher z-transform
                se_z = 1.0 / torch.sqrt(n_val - 3)

                # Test statistic under alternative hypothesis
                z_stat = z_r / se_z

                # Critical values using standard normal approximation
                sqrt_2 = math.sqrt(2.0)

                if self.alternative == "two-sided":
                    z_alpha_2 = (
                        torch.erfinv(torch.tensor(1 - alpha / 2, dtype=r_val.dtype))
                        * sqrt_2
                    )
                    cdf_upper = 0.5 * (1 + torch.erf((z_alpha_2 - z_stat) / sqrt_2))
                    cdf_lower = 0.5 * (1 + torch.erf((-z_alpha_2 - z_stat) / sqrt_2))
                    power_val = 1 - (cdf_upper - cdf_lower)
                elif self.alternative == "greater":
                    z_alpha = (
                        torch.erfinv(torch.tensor(1 - alpha, dtype=r_val.dtype))
                        * sqrt_2
                    )
                    power_val = 1 - 0.5 * (1 + torch.erf((z_alpha - z_stat) / sqrt_2))
                elif self.alternative == "less":
                    z_alpha = (
                        torch.erfinv(torch.tensor(alpha, dtype=r_val.dtype)) * sqrt_2
                    )
                    power_val = 0.5 * (1 + torch.erf((z_alpha - z_stat) / sqrt_2))

                power_val = torch.clamp(power_val, 0.0, 1.0)
                y_values.append(float(power_val))

            # Plot line with label
            label = f"{param_name} = {param_val}"
            ax.plot(x_values, y_values, label=label, **kwargs)

        # Customize plot
        ax.set_xlabel(x_label)
        ax.set_ylabel("Statistical Power")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if title is None:
            title = f"Power Analysis: Correlation Test ({self.alternative})"
        ax.set_title(title)

        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, "
            f"alternative={self.alternative!r})"
        )
