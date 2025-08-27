import torch
from torch import Tensor


class StandardNormal(torch.distributions.Normal):
    r"""
    Standard normal distribution N(0, 1) with inverse cumulative distribution function.

    This is a convenience class that pre-configures the normal distribution with
    mean=0 and std=1, which is commonly used in statistical hypothesis testing
    and power analysis.

    Parameters
    ----------
    validate_args : bool, optional
        Whether to validate arguments.

    Examples
    --------
    >>> import torch
    >>> from beignet.distributions import StandardNormal
    >>> dist = StandardNormal()
    >>> # Get critical values for hypothesis testing
    >>> z_alpha = dist.icdf(torch.tensor(0.975))  # 95% confidence
    >>> z_alpha
    tensor(1.9600)
    >>>
    >>> # Compute power calculations
    >>> power = 1 - dist.cdf(z_alpha - torch.tensor(2.0))  # Effect size 2
    >>> power
    tensor(0.5199)

    Notes
    -----
    The standard normal distribution is fundamental to statistical inference.
    It has mean μ = 0, variance σ² = 1, and is the basis for Z-tests,
    confidence intervals, and power calculations in many statistical methods.

    This class provides the same functionality as Normal(0, 1) but with
    a more convenient interface that doesn't require specifying the parameters.
    """

    def __init__(self, validate_args: bool | None = None):
        """
        Initialize StandardNormal distribution.

        Parameters
        ----------
        validate_args : bool | None, optional
            Whether to validate arguments.
        """
        # Pre-configure as N(0, 1) - use float32 as default to match other distributions
        loc = torch.tensor(0.0, dtype=torch.float32)
        scale = torch.tensor(1.0, dtype=torch.float32)
        super().__init__(loc, scale, validate_args=validate_args)

    def to(self, dtype_or_device):
        """Move distribution to specified dtype or device."""
        if isinstance(dtype_or_device, torch.dtype):
            # Handle dtype conversion
            new_loc = self.loc.to(dtype_or_device)
            new_scale = self.scale.to(dtype_or_device)
            return StandardNormal.__new__(
                StandardNormal,
            ).__init__(validate_args=self._validate_args)
        else:
            # Handle device conversion
            return super().to(dtype_or_device)

    @classmethod
    def from_dtype(cls, dtype: torch.dtype, validate_args: bool | None = None):
        """
        Create StandardNormal distribution with specified dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Desired dtype for the distribution parameters.
        validate_args : bool | None, optional
            Whether to validate arguments.

        Returns
        -------
        StandardNormal
            Standard normal distribution with specified dtype.
        """
        instance = cls.__new__(cls)
        loc = torch.tensor(0.0, dtype=dtype)
        scale = torch.tensor(1.0, dtype=dtype)
        super(StandardNormal, instance).__init__(
            loc, scale, validate_args=validate_args
        )
        return instance

    def icdf(self, value: Tensor) -> Tensor:
        r"""
        Inverse cumulative distribution function (quantile function).

        For the standard normal distribution, this computes the quantiles
        using the error function inverse: Φ⁻¹(p) = √2 * erfinv(2p - 1)

        Parameters
        ----------
        value : Tensor
            Probability values in [0, 1].

        Returns
        -------
        Tensor
            Quantiles corresponding to the given probabilities.

        Examples
        --------
        >>> dist = StandardNormal()
        >>> # Common critical values
        >>> dist.icdf(torch.tensor(0.975))  # 95% confidence (two-tailed)
        tensor(1.9600)
        >>> dist.icdf(torch.tensor(0.95))   # 90% confidence (one-tailed)
        tensor(1.6449)
        """
        return super().icdf(value)

    def cdf(self, value: Tensor) -> Tensor:
        r"""
        Cumulative distribution function.

        For the standard normal distribution, this is the standard normal CDF:
        Φ(z) = (1 + erf(z/√2)) / 2

        Parameters
        ----------
        value : Tensor
            Values at which to evaluate the CDF.

        Returns
        -------
        Tensor
            CDF values corresponding to the input values.
        """
        return super().cdf(value)

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the distribution parameters."""
        return self.loc.dtype

    def __repr__(self) -> str:
        return "StandardNormal()"
