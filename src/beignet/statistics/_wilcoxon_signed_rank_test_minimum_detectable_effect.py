import math

import torch
from torch import Tensor

from ._wilcoxon_signed_rank_test_power import wilcoxon_signed_rank_test_power


def wilcoxon_signed_rank_test_minimum_detectable_effect(
    nobs: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Minimum detectable prob_positive = P(D>0) + 0.5 P(D=0) for Wilcoxon signed-rank test.

    Parameters
    ----------
    nobs : Tensor
        Number of non-zero paired differences.
    power : float, default=0.8
        Target power.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}
        Direction; returns ≥0.5 for "greater"/"two-sided" and ≤0.5 for "less".

    Returns
    -------
    Tensor
        Minimal detectable prob_positive value.

    When to Use
    -----------
    **Traditional Statistics:**

    - **Paired non-parametric analysis:** When paired differences are non-normal or ordinal
    - **Median-based comparisons:** Testing shifts in paired data medians
    - **Robust paired testing:** Alternative to paired t-test when outliers are expected
    - **Small sample paired studies:** When sample sizes are too small for normality
    - **Skewed difference distributions:** When paired differences are heavily skewed
    - **Ordinal paired data:** Analyzing paired ratings, scores, or ranked measurements

    **Machine Learning Applications:**

    - **Model performance comparison (paired):** Minimum detectable improvements using same test sets
    - **Algorithm robustness testing:** Paired comparison of performance under different conditions
    - **Cross-validation stability:** Testing consistency of model performance across paired CV folds
    - **Hyperparameter impact assessment:** Paired evaluation of parameter changes on same data
    - **Feature engineering validation:** Before/after comparison of feature modifications
    - **Data augmentation evaluation:** Paired testing of augmentation effects on model performance
    - **Transfer learning assessment:** Comparing performance on paired source/target domain samples
    - **Ensemble component analysis:** Paired evaluation of individual ensemble member contributions
    - **Active learning effectiveness:** Paired comparison of learning progress with/without active selection
    - **Regularization impact testing:** Paired evaluation of regularization effects on same datasets
    - **Preprocessing pipeline comparison:** Paired testing of different data preprocessing approaches
    - **Model interpretability validation:** Paired comparison of explanation quality metrics
    - **Fairness testing:** Paired evaluation of model performance across demographic groups
    - **Temporal model stability:** Paired comparison of model performance across time periods
    - **Multi-task learning evaluation:** Paired assessment of shared vs. separate task components

    **Interpretation Guidelines:**

    - **prob_positive = 0.5:** No difference (random pairing)
    - **prob_positive = 0.6:** Small effect (60% positive differences)
    - **prob_positive = 0.7:** Medium effect (70% positive differences)
    - **prob_positive = 0.8:** Large effect (80% positive differences)
    - **Consider practical significance:** Statistical detectability should align with meaningful differences
    - **Robust to outliers:** Less sensitive to extreme values than parametric tests
    """
    sample_size_0 = torch.as_tensor(nobs)
    scalar_out = sample_size_0.ndim == 0
    sample_size = torch.atleast_1d(sample_size_0)
    dtype = torch.float64 if sample_size.dtype == torch.float64 else torch.float32
    sample_size = torch.clamp(sample_size.to(dtype), min=5.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    S = sample_size * (sample_size + 1.0) / 2.0
    var0 = sample_size * (sample_size + 1.0) * (2.0 * sample_size + 1.0) / 24.0
    sd0 = torch.sqrt(torch.clamp(var0, min=1e-12))

    sqrt2 = math.sqrt(2.0)

    def z_of(prob: float) -> Tensor:
        q = torch.tensor(prob, dtype=dtype)
        eps = torch.finfo(dtype).eps
        q = torch.clamp(q, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * q - 1.0)

    z_alpha = z_of(1 - alpha / 2) if alt == "two-sided" else z_of(1 - alpha)
    z_beta = z_of(power)
    delta = (z_alpha + z_beta) * sd0 / torch.clamp(S, min=1e-12)

    if alt == "less":
        prob_initial = torch.clamp(0.5 - delta, 0.0, 1.0)
    else:
        prob_initial = torch.clamp(0.5 + delta, 0.0, 1.0)

    if alt == "less":
        prob_lo = torch.zeros_like(prob_initial)
        prob_hi = torch.full_like(prob_initial, 0.5)
    else:
        prob_lo = torch.full_like(prob_initial, 0.5)
        prob_hi = torch.ones_like(prob_initial)

    if alt == "less":
        max_power_prob = prob_lo
    else:
        max_power_prob = prob_hi

    max_power = wilcoxon_signed_rank_test_power(
        max_power_prob, sample_size, alpha=alpha, alternative=alt
    )

    unattainable = max_power < power - 1e-6

    probability = (prob_lo + prob_hi) * 0.5
    for _ in range(24):
        current_power = wilcoxon_signed_rank_test_power(
            probability, sample_size, alpha=alpha, alternative=alt
        )
        too_low = current_power < power

        if alt == "less":
            prob_hi = torch.where(too_low, probability, prob_hi)
            prob_lo = torch.where(too_low, prob_lo, probability)
        else:
            prob_lo = torch.where(too_low, probability, prob_lo)
            prob_hi = torch.where(too_low, prob_hi, probability)
        probability = (prob_lo + prob_hi) * 0.5

    probability = torch.where(unattainable, max_power_prob, probability)

    if scalar_out:
        probability_s = probability.reshape(())
        if out is not None:
            out.copy_(probability_s)
            return out
        return probability_s
    else:
        if out is not None:
            out.copy_(probability)
            return out
        return probability
