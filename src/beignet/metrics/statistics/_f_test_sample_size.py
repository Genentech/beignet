"""F-test sample size metric."""

from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

from beignet.statistics._f_test_sample_size import f_test_sample_size


class FTestSampleSize(Metric):
    """Computes required sample size for general F-tests.

    Parameters
    ----------
    effect_size : Tensor
        Effect size (Cohen's f² or similar). This represents the magnitude of
        the effect being tested. For regression contexts, this could be the
        R² change or similar measure.
    df1 : Tensor
        Degrees of freedom for the numerator (effect).
    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> metric = FTestSampleSize()
    >>> effect_size = torch.tensor(0.15)
    >>> df1 = torch.tensor(3)
    >>> metric.update(effect_size, df1)
    >>> metric.compute()
    tensor(100)
    """

    full_state_update: bool = False

    def __init__(self, power: float = 0.8, alpha: float = 0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.power = power
        self.alpha = alpha
        self.add_state("effect_size", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("df1", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, effect_size: Tensor, df1: Tensor) -> None:
        """Update the metric state.

        Parameters
        ----------
        effect_size : Tensor
            Effect size (Cohen's f² or similar).
        df1 : Tensor
            Degrees of freedom for numerator.
        """
        if self.total == 0:
            self.effect_size = effect_size
            self.df1 = df1
        else:
            self.effect_size = (self.effect_size * self.total + effect_size) / (
                self.total + 1
            )
            self.df1 = (self.df1 * self.total + df1) / (self.total + 1)
        self.total += 1

    def compute(self) -> Tensor:
        """Compute the required sample size."""
        return f_test_sample_size(
            effect_size=self.effect_size,
            df1=self.df1,
            power=self.power,
            alpha=self.alpha,
        )

    def plot(
        self,
        dep_var: str = "effect_size",
        effect_size: Optional[Any] = None,
        df1: Optional[Any] = None,
        power: Optional[Any] = None,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot sample size curves varying different parameters.

        Parameters
        ----------
        dep_var : str, default="effect_size"
            The variable to plot on x-axis. Either "effect_size", "df1", or "power".
        effect_size : array-like, optional
            Range of effect sizes to plot. If None, uses reasonable default range.
        df1 : array-like, optional
            Range of numerator degrees of freedom to plot. If None, uses reasonable default range.
        power : array-like, optional
            Range of power values to plot. If None, uses reasonable default range.
        alpha : float, default=0.05
            Significance level for sample size calculation.
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
        >>> metric = FTestSampleSize()
        >>> fig = metric.plot(dep_var="effect_size", df1=[2, 4, 6])
        >>> fig = metric.plot(dep_var="power", effect_size=[0.1, 0.25, 0.4])
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
        if dep_var == "effect_size":
            if effect_size is None:
                x_values = np.linspace(0.01, 0.5, 100)
            else:
                x_values = np.asarray(effect_size)

            if df1 is None:
                param_values = [2, 4, 6]  # Different numerator df
                param_name = "df1"
            else:
                param_values = np.asarray(df1)
                param_name = "df1"

            x_label = "Effect Size (Cohen's f²)"

        elif dep_var == "df1":
            if df1 is None:
                x_values = np.linspace(1, 20, 100)
            else:
                x_values = np.asarray(df1)

            if effect_size is None:
                param_values = [0.1, 0.25, 0.4]  # Small, medium, large effects
                param_name = "Effect Size"
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"

            x_label = "Numerator Degrees of Freedom (df1)"

        elif dep_var == "power":
            if power is None:
                x_values = np.linspace(0.5, 0.95, 100)
            else:
                x_values = np.asarray(power)

            if effect_size is None:
                param_values = [0.1, 0.25, 0.4]
                param_name = "Effect Size"
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"

            x_label = "Statistical Power"
        else:
            raise ValueError(
                f"dep_var must be 'effect_size', 'df1', or 'power', got {dep_var}"
            )

        # Plot sample size curves for different parameter values
        for param_val in param_values:
            y_values = []

            for x_val in x_values:
                if dep_var == "effect_size":
                    sample_size_val = f_test_sample_size(
                        effect_size=torch.tensor(float(x_val)),
                        df1=torch.tensor(float(param_val)),
                        power=self.power,
                        alpha=alpha,
                    )
                elif dep_var == "df1":
                    sample_size_val = f_test_sample_size(
                        effect_size=torch.tensor(float(param_val)),
                        df1=torch.tensor(float(x_val)),
                        power=self.power,
                        alpha=alpha,
                    )
                elif dep_var == "power":
                    sample_size_val = f_test_sample_size(
                        effect_size=torch.tensor(float(param_val)),
                        df1=torch.tensor(3.0),  # Default df1
                        power=torch.tensor(float(x_val)),
                        alpha=alpha,
                    )

                y_values.append(float(sample_size_val))

            # Plot line with label
            label = f"{param_name} = {param_val}"
            ax.plot(x_values, y_values, label=label, **kwargs)

        # Customize plot
        ax.set_xlabel(x_label)
        ax.set_ylabel("Required Sample Size")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if title is None:
            title = "Sample Size Analysis: F-Test"
        ax.set_title(title)

        plt.tight_layout()
        return fig
