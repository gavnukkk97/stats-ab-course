"""Small, inspectable helpers for M6 examples; assumptions remain part of the design."""
import numpy as np
from scipy import stats


def descendants(edges, node):
    found, stack = set(), [node]
    while stack:
        parent = stack.pop()
        for a, b in edges:
            if a == parent and b not in found:
                found.add(b)
                stack.append(b)
    return found


def path_blocked(edges, path, conditioned):
    """d-separation rule for an undirected path in a DAG."""
    es, conditioned = set(edges), set(conditioned)
    for a, m, b in zip(path, path[1:], path[2:]):
        collider = (a, m) in es and (b, m) in es
        if collider and not (({m} | descendants(edges, m)) & conditioned):
            return True
        if not collider and m in conditioned:
            return True
    return False


def rosenbaum_upper_p(wins, n_pairs, gamma):
    """Upper one-sided sign-test p bound for a prespecified positive effect.

    Requires disjoint matched pairs, a sharp null, and Rosenbaum's odds bound.
    Gamma is an upper bound, not a known assignment probability.
    """
    if gamma < 1 or n_pairs < 1 or not 0 <= wins <= n_pairs:
        raise ValueError("Require gamma >= 1 and 0 <= wins <= n_pairs, n_pairs > 0")
    return float(stats.binom.sf(wins - 1, n_pairs, gamma / (1 + gamma)))


def evalue_point(rr):
    if rr <= 0:
        raise ValueError("Risk ratio must be positive")
    r = max(rr, 1 / rr)
    return float(r + np.sqrt(r * (r - 1)))


def evalue_interval(rr, lo, hi):
    """Point E-value and E-value of the confidence limit nearest RR=1."""
    if not 0 < lo <= rr <= hi:
        raise ValueError("Require 0 < lower <= estimate <= upper")
    bound = 1.0 if lo <= 1 <= hi else evalue_point(lo if rr > 1 else hi)
    return evalue_point(rr), bound


def paired_rr_interval(treated, control, alpha=0.05):
    """Large-sample log-RR interval with covariance within disjoint pairs.

    Requires independent pairs and nonzero risk in both arms. This is not an
    exact small-sample interval or an inference procedure for reused controls.
    """
    t, c = np.asarray(treated, float), np.asarray(control, float)
    if t.shape != c.shape or len(t) < 2:
        raise ValueError("Need equally sized arrays with at least two pairs")
    pt, pc = t.mean(), c.mean()
    if min(pt, pc) <= 0:
        raise ValueError("A zero risk needs a different interval method")
    rr = pt / pc
    influence = t / pt - c / pc
    se = np.std(influence, ddof=1) / np.sqrt(len(t))
    width = stats.norm.ppf(1 - alpha / 2) * se
    return float(rr), float(rr * np.exp(-width)), float(rr * np.exp(width))


def effective_size(weights):
    w = np.asarray(weights, float)
    if not np.any(w):
        return 0.0
    return float(w.sum() ** 2 / np.sum(w ** 2))


def matched_smd(x, treatment, idx_t, idx_c):
    """Matched weighted means; standardization fixed at the original pooled SD."""
    x, d = np.asarray(x), np.asarray(treatment)
    scale = np.sqrt((x[d == 1].var(ddof=1) + x[d == 0].var(ddof=1)) / 2)
    diff = x[idx_t].mean() - x[idx_c].mean()
    return float(diff / scale) if scale > 0 else (0.0 if diff == 0 else np.inf)


def cluster_covariance(y, X, clusters):
    """CR1 covariance; approximate inference requires enough independent clusters."""
    y, X, clusters = np.asarray(y), np.asarray(X), np.asarray(clusters)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    labels = np.unique(clusters)
    n, k = X.shape[0], np.linalg.matrix_rank(X)
    if len(labels) < 2 or n <= k:
        raise ValueError("Insufficient clusters or residual degrees of freedom")
    bread = np.linalg.pinv(X.T @ X)
    scores = np.stack([X[clusters == g].T @ residual[clusters == g] for g in labels])
    correction = len(labels) / (len(labels) - 1) * (n - 1) / (n - k)
    return beta, correction * bread @ (scores.T @ scores) @ bread


def exact_pairs(X, treatment, seed=0):
    """Disjoint exact pairs on genuinely discrete covariates, without coarsening."""
    X, d = np.asarray(X), np.asarray(treatment)
    rng, it, ic = np.random.default_rng(seed), [], []
    _, labels = np.unique(X, axis=0, return_inverse=True)
    for label in np.unique(labels):
        t = rng.permutation(np.flatnonzero((labels == label) & (d == 1)))
        c = rng.permutation(np.flatnonzero((labels == label) & (d == 0)))
        k = min(len(t), len(c))
        it.extend(t[:k]); ic.extend(c[:k])
    if not it:
        raise ValueError("No exact pairs: this design cannot answer the question")
    return np.asarray(it), np.asarray(ic)


def method_options(can_randomize=False, blocked_by_ethics=False, already_happened=False,
                   has_control_group=False, parallel_trends=False, has_threshold=False,
                   single_unit=False, rich_covariates=False, has_instrument=False,
                   has_preperiod=False, has_donors=False, has_control_series=False,
                   plausible_its=False):
    """Candidate designs, not a ranking of truth. Flags denote justified assumptions."""
    if can_randomize and not already_happened and not blocked_by_ethics:
        return ["A/B: выбрать estimand, единицу назначения и учесть интерференцию"]
    options = []
    if has_threshold:
        options.append("RDD: непрерывность потенциальных исходов у порога; локальный estimand")
    if has_instrument:
        options.append("IV: обосновать independence/exclusion/monotonicity; LATE и первый этап")
    if has_control_group and has_preperiod and parallel_trends:
        options.append("DiD: параллельная контрфактическая динамика, no anticipation; ATT")
    if single_unit and has_preperiod and has_donors:
        options.append("Synthetic control: незатронутые доноры и качество pre-fit")
    if has_preperiod and has_control_series:
        options.append("CausalImpact: незатронутые ряды и стабильная прогнозная связь")
    if has_control_group and rich_covariates:
        options.append("Adjustment: matching/IPW/DR при exchangeability и overlap; выбрать estimand")
    if has_preperiod and plausible_its:
        options.append("ITS: явно принять экстраполяцию и отсутствие одновременных шоков")
    return options or ["Идентификация не обоснована: уточнить дизайн и собрать нужные данные"]


def crossfit_binary_demo(seed=6440, n=20000):
    """Saturated nuisance models on binary X illustrate PLR's overlap estimand."""
    rng = np.random.default_rng(seed)
    x = rng.binomial(1, .5, n)
    propensity = np.where(x == 0, .05, .5)
    effect = -3 + 6*x
    d = rng.binomial(1, propensity)
    y = 2 + 2*x + effect*d + rng.normal(size=n)
    fold = np.arange(n) % 4
    rng.shuffle(fold)
    m, l, mu0, mu1 = [np.empty(n) for _ in range(4)]
    for k in range(4):
        for value in [0, 1]:
            train, test = (fold != k) & (x == value), (fold == k) & (x == value)
            m[test], l[test] = d[train].mean(), y[train].mean()
            mu0[test] = y[train & (d == 0)].mean()
            mu1[test] = y[train & (d == 1)].mean()
    dr, yr = d-m, y-l
    plr = dr @ yr / (dr @ dr)
    aipw = np.mean(mu1-mu0 + d*(y-mu1)/m - (1-d)*(y-mu0)/(1-m))
    overlap_truth = np.mean(propensity*(1-propensity)*effect) / np.mean(propensity*(1-propensity))
    return dict(plr=float(plr), aipw=float(aipw), ate=float(effect.mean()),
                overlap_truth=float(overlap_truth))
