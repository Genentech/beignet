import torch
from torch import Tensor


class Normal(torch.distributions.Normal):
    r"""
    Normal distribution with consistent interface for statistical computations.

    Extends torch.distributions.Normal to provide a consistent interface
    matching other beignet.distributions classes. Since torch.distributions.Normal
    already has icdf implemented, this serves as a compatibility wrapper.

    Parameters
    ----------
    loc : Tensor
        Mean of the distribution.
    scale : Tensor
        Standard deviation of the distribution.

    Examples
    --------
    >>> import torch
    >>> from beignet.distributions import Normal
    >>> dist = Normal(torch.tensor(0.0), torch.tensor(1.0))
    >>> quantile = dist.icdf(torch.tensor(0.975))
    >>> quantile
    tensor(1.9600)
    """

    def __init__(self, loc: Tensor, scale: Tensor, validate_args: bool | None = None):
        """
        Initialize Normal distribution.

        Parameters
        ----------
        loc : Tensor
            Mean of the distribution.
        scale : Tensor
            Standard deviation of the distribution.
        validate_args : bool | None, optional
            Whether to validate arguments.
        """
        super().__init__(loc, scale, validate_args=validate_args)

    def icdf(self, value: Tensor) -> Tensor:
        r"""
        Inverse cumulative distribution function (quantile function).

        This method is inherited from torch.distributions.Normal and provides
        exact normal quantiles using the inverse error function.

        Parameters
        ----------
        value : Tensor
            Probability values in [0, 1].

        Returns
        -------
        Tensor
            Quantiles corresponding to the given probabilities.

        Notes
        -----
        The implementation uses the relationship:

        .. math::
            \Phi^{-1}(p) = \mu + \sigma \sqrt{2} \text{erfinv}(2p - 1)

        where :math:`\Phi^{-1}` is the inverse standard normal CDF,
        :math:`\mu` is the mean, :math:`\sigma` is the standard deviation.
        """
        return super().icdf(value)
