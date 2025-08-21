"""F-test power metric."""

from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

from beignet.statistics._f_test_power import f_test_power


class FTestPower(Metric):
    """Computes statistical power for general F-tests.

    Parameters
    ----------
    effect_size : Tensor
        Effect size (Cohen's f² or similar). This represents the magnitude of
        the effect being tested. For regression contexts, this could be the
        R² change or similar measure.
    df1 : Tensor
        Degrees of freedom for the numerator (effect).
    df2 : Tensor
        Degrees of freedom for the denominator (error).
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Examples
    --------
    >>> metric = FTestPower()
    >>> effect_size = torch.tensor(0.15)
    >>> df1 = torch.tensor(3)
    >>> df2 = torch.tensor(96)
    >>> metric.update(effect_size, df1, df2)
    >>> metric.compute()
    tensor(0.8743)
    """

    full_state_update: bool = False

    def __init__(self, alpha: float = 0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.add_state("effect_size", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("df1", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("df2", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, effect_size: Tensor, df1: Tensor, df2: Tensor) -> None:
        """Update the metric state.

        Parameters
        ----------
        effect_size : Tensor
            Effect size (Cohen's f² or similar).
        df1 : Tensor
            Degrees of freedom for numerator.
        df2 : Tensor
            Degrees of freedom for denominator.
        """
        if self.total == 0:
            self.effect_size = effect_size
            self.df1 = df1
            self.df2 = df2
        else:
            self.effect_size = (self.effect_size * self.total + effect_size) / (
                self.total + 1
            )
            self.df1 = (self.df1 * self.total + df1) / (self.total + 1)
            self.df2 = (self.df2 * self.total + df2) / (self.total + 1)
        self.total += 1

    def compute(self) -> Tensor:
        """Compute the statistical power."""
        return f_test_power(
            effect_size=self.effect_size,
            df1=self.df1,
            df2=self.df2,
            alpha=self.alpha,
        )

    def plot(
        self,
        dep_var: str = "effect_size",
        effect_size: Optional[Any] = None,
        df1: Optional[Any] = None,
        df2: Optional[Any] = None,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot power curves varying different parameters.

        Parameters
        ----------
        dep_var : str, default="effect_size"
            The variable to plot on x-axis. Either "effect_size", "df1", or "df2".
        effect_size : array-like, optional
            Range of effect sizes to plot. If None, uses reasonable default range.
        df1 : array-like, optional
            Range of numerator degrees of freedom to plot. If None, uses reasonable default range.
        df2 : array-like, optional
            Range of denominator degrees of freedom to plot. If None, uses reasonable default range.
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
        >>> metric = FTestPower()
        >>> fig = metric.plot(dep_var="effect_size", df1=[2, 4, 6])
        >>> fig = metric.plot(dep_var="df2", effect_size=[0.1, 0.25, 0.4])
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            ) from None

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
                default_df2 = 50
            else:
                param_values = np.asarray(df1)
                param_name = "df1"
                default_df2 = 50

            x_label = "Effect Size (Cohen's f²)"

        elif dep_var == "df1":
            if df1 is None:
                x_values = np.linspace(1, 20, 100)
            else:
                x_values = np.asarray(df1)

            if effect_size is None:
                param_values = [0.1, 0.25, 0.4]  # Small, medium, large effects
                param_name = "Effect Size"
                default_df2 = 50
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"
                default_df2 = 50

            x_label = "Numerator Degrees of Freedom (df1)"

        elif dep_var == "df2":
            if df2 is None:
                x_values = np.linspace(10, 200, 100)
            else:
                x_values = np.asarray(df2)

            if effect_size is None:
                param_values = [0.1, 0.25, 0.4]
                param_name = "Effect Size"
                default_df1 = 3
            else:
                param_values = np.asarray(effect_size)
                param_name = "Effect Size"
                default_df1 = 3

            x_label = "Denominator Degrees of Freedom (df2)"
        else:
            raise ValueError(
                f"dep_var must be 'effect_size', 'df1', or 'df2', got {dep_var}"
            )

        # Plot power curves for different parameter values
        for param_val in param_values:
            y_values = []

            for x_val in x_values:
                if dep_var == "effect_size":
                    power_val = f_test_power(
                        effect_size=torch.tensor(float(x_val)),
                        df1=torch.tensor(float(param_val)),
                        df2=torch.tensor(float(default_df2)),
                        alpha=alpha,
                    )
                elif dep_var == "df1":
                    power_val = f_test_power(
                        effect_size=torch.tensor(float(param_val)),
                        df1=torch.tensor(float(x_val)),
                        df2=torch.tensor(float(default_df2)),
                        alpha=alpha,
                    )
                elif dep_var == "df2":
                    power_val = f_test_power(
                        effect_size=torch.tensor(float(param_val)),
                        df1=torch.tensor(float(default_df1)),
                        df2=torch.tensor(float(x_val)),
                        alpha=alpha,
                    )

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
            title = "Power Analysis: F-Test"
        ax.set_title(title)

        plt.tight_layout()
        return fig
