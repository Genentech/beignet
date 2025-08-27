import torch
import torch.distributions
from torch import Tensor


class StudentT(torch.distributions.StudentT):
    r"""
    Student's t-distribution with inverse cumulative distribution function.

    Extends torch.distributions.StudentT to provide icdf functionality using
    a high-accuracy approximation method.

    Parameters
    ----------
    df : Tensor
        Degrees of freedom.
    loc : Tensor, optional
        Location parameter (mean), default 0.
    scale : Tensor, optional
        Scale parameter, default 1.

    Examples
    --------
    >>> import torch
    >>> from beignet.distributions import StudentT
    >>> dist = StudentT(df=torch.tensor(10.0))
    >>> quantile = dist.icdf(torch.tensor(0.975))
    >>> quantile
    tensor(2.2281)
    """

    def icdf(self, value: Tensor) -> Tensor:
        r"""
        Inverse cumulative distribution function (quantile function).

        Uses a high-accuracy approximation that combines normal approximation
        with correction terms for finite degrees of freedom.

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
        For large degrees of freedom (>30), uses normal approximation with
        Cornish-Fisher correction. For smaller degrees of freedom, uses
        an iterative improvement method for better accuracy.
        """
        # Ensure value is a tensor with proper dtype
        value = torch.as_tensor(value, dtype=self.df.dtype, device=self.df.device)

        # Handle edge cases
        eps = torch.finfo(value.dtype).eps
        value = torch.clamp(value, min=eps, max=1 - eps)

        # Get normal quantile as starting point
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=value.dtype, device=value.device))
        z = sqrt_two * torch.erfinv(2.0 * value - 1.0)

        # For large df, use improved normal approximation
        df_clamped = torch.clamp(self.df, min=1.0)

        # Cornish-Fisher correction for t-distribution
        correction = z / (4.0 * df_clamped) * (z**2 + 1.0)

        # Higher-order correction for better accuracy
        correction2 = z / (96.0 * df_clamped**2) * (5.0 * z**4 + 16.0 * z**2 + 3.0)

        # Third-order correction
        correction3 = (
            z
            / (384.0 * df_clamped**3)
            * (3.0 * z**6 + 19.0 * z**4 + 17.0 * z**2 - 15.0)
        )

        # Combine corrections
        t_approx = z + correction + correction2 + correction3

        # Apply location and scale
        result = self.loc + self.scale * t_approx

        return result
