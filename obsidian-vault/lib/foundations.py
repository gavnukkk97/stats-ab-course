"""Small, tested calculations shared by the foundations examples.

No simulation or file writes happen on import. Array arguments are supported.
"""
import numpy as np
from scipy import stats


def normal_power(effect, se, alpha=0.05):
    """Exact two-sided Gaussian rejection probability for a given true effect."""
    delta = np.abs(np.asarray(effect)) / np.asarray(se)
    critical = stats.norm.ppf(1 - alpha / 2)
    return stats.norm.cdf(delta - critical) + stats.norm.cdf(-delta - critical)


def coin_pvalues(n=10):
    return np.array([stats.binomtest(k, n, 0.5).pvalue for k in range(n + 1)])


def permutation_differences(a, b, rng, repeats=5000):
    """One partition per draw; valid randomization null requires exchangeability."""
    pool = np.concatenate([a, b])
    n = len(a)
    result = np.empty(repeats)
    for i in range(repeats):
        permuted = rng.permutation(pool)
        result[i] = permuted[n:].mean() - permuted[:n].mean()
    return result


def permutation_pvalue(observed, draws):
    draws = np.asarray(draws)
    return (np.count_nonzero(np.abs(draws) >= abs(observed)) + 1) / (len(draws) + 1)


def holm_thresholds(m, alpha=0.05):
    return alpha / np.arange(m, 0, -1)


def type_m(estimates, true_effect):
    """Magnitude exaggeration among the supplied significant estimates."""
    if true_effect == 0:
        raise ValueError("Type M is undefined for zero true effect")
    return np.mean(np.abs(estimates)) / abs(true_effect)


def mc_interval(successes, trials, confidence=0.95):
    """Wilson interval for a binomial simulation rate, not for treatment effect."""
    z = stats.norm.ppf((1 + confidence) / 2)
    p = np.asarray(successes) / trials
    center = (p + z*z/(2*trials)) / (1 + z*z/trials)
    half = z*np.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / (1 + z*z/trials)
    return center-half, center+half


def mc_report(successes, trials):
    low, high = mc_interval(successes, trials)
    return f"{successes/trials:.2%} (MC 95% CI [{low:.2%}; {high:.2%}], B={trials})"
