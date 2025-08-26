import torch
from torch import Tensor

from ._f_test_power import f_test_power


def f_test_minimum_detectable_effect(
    df1: Tensor,
    df2: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Minimum detectable effect for general F-tests (returns minimal f²).

    Parameters
    ----------
    df1 : Tensor
        Numerator degrees of freedom.
    df2 : Tensor
        Denominator degrees of freedom.
    power : float, default=0.8
        Target power.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    Tensor
        Minimal f² achieving the requested power.

    When to Use
    -----------
    **Traditional Statistics:**

    - **Variance ratio testing:** Comparing variances between groups or conditions
    - **Model comparison:** Testing if additional terms significantly improve model fit
    - **Nested model testing:** Comparing restricted vs. unrestricted statistical models
    - **Regression analysis:** Testing overall model significance or subset of predictors
    - **Design of experiments:** Planning studies with known degrees of freedom
    - **Quality control:** Testing variance homogeneity in manufacturing processes

    **Machine Learning Applications:**

    - **Feature subset evaluation:** Minimum detectable improvement from additional features
    - **Model complexity testing:** Comparing models with different numbers of parameters
    - **Cross-validation analysis:** Testing significance of performance differences across folds
    - **Hyperparameter optimization:** Minimum detectable effects of parameter changes
    - **Ensemble method comparison:** Testing variance differences between ensemble components
    - **Regularization effect analysis:** Detecting minimum impacts of regularization parameters
    - **Architecture comparison:** Testing performance differences between neural network architectures
    - **Data augmentation validation:** Minimum detectable effects of data augmentation techniques
    - **Transfer learning assessment:** Testing significance of domain adaptation improvements
    - **Active learning evaluation:** Minimum detectable differences in learning curve slopes
    - **Distributed training analysis:** Testing consistency across different training configurations
    - **Batch size optimization:** Detecting minimum effects of batch size on model variance
    - **Learning rate scheduling:** Minimum detectable impacts of learning rate decay strategies
    - **Multi-task learning:** Testing shared vs. separate model components effectiveness
    - **Federated learning evaluation:** Minimum detectable differences across federated clients

    **Interpretation Guidelines:**

    - **f² = 0.02:** Small effect (Cohen's convention)
    - **f² = 0.15:** Medium effect
    - **f² = 0.35:** Large effect
    - **Values < 0.02:** Very small effects, may lack practical significance
    - **Consider degrees of freedom:** Higher df1/df2 ratios require larger effect sizes
    - **F-statistic relationship:** F = 1 + f² under null, so f² represents proportional variance increase
    """
    df1_0 = torch.as_tensor(df1)
    df2_0 = torch.as_tensor(df2)
    scalar_out = df1_0.ndim == 0 and df2_0.ndim == 0
    df1 = torch.atleast_1d(df1_0)
    df2 = torch.atleast_1d(df2_0)
    dtype = (
        torch.float64
        if (df1.dtype == torch.float64 or df2.dtype == torch.float64)
        else torch.float32
    )
    df1 = torch.clamp(df1.to(dtype), min=1.0)
    df2 = torch.clamp(df2.to(dtype), min=1.0)

    # Initial guess based on inverse of sample size approximation in _f_test_sample_size
    # base_n ≈ ((zα+zβ)/sqrt(f²))²; with N ≈ df1+df2+1 => f² ≈ ((zα+zβ)/sqrt(N))²
    N = df1 + df2 + 1.0
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * torch.sqrt(
        torch.tensor(2.0, dtype=dtype)
    )
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * torch.sqrt(
        torch.tensor(2.0, dtype=dtype)
    )
    f2_0 = torch.clamp(((z_alpha + z_beta) / torch.sqrt(N)) ** 2, min=1e-8)

    f2_lo = torch.zeros_like(f2_0) + 1e-8
    f2_hi = torch.clamp(2.0 * f2_0 + 1e-6, min=1e-6)

    # Ensure upper bound is sufficient
    for _ in range(8):
        p_hi = f_test_power(f2_hi, df1, df2, alpha)
        need_expand = p_hi < power
        if not torch.any(need_expand):
            break
        f2_hi = torch.where(need_expand, f2_hi * 2.0, f2_hi)
        f2_hi = torch.clamp(f2_hi, max=torch.tensor(10.0, dtype=dtype))

    # Bisection
    x = (f2_lo + f2_hi) * 0.5
    for _ in range(24):
        p_mid = f_test_power(x, df1, df2, alpha)
        go_right = p_mid < power
        f2_lo = torch.where(go_right, x, f2_lo)
        f2_hi = torch.where(go_right, f2_hi, x)
        x = (f2_lo + f2_hi) * 0.5

    out_t = torch.clamp(x, min=0.0)
    if scalar_out:
        out_scalar = out_t.reshape(())
        if out is not None:
            out.copy_(out_scalar)
            return out
        return out_scalar
    else:
        if out is not None:
            out.copy_(out_t)
            return out
        return out_t
