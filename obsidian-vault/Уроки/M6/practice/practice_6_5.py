# -*- coding: utf-8 -*-
"""M6.5: diagnostics and sensitivity in an explicitly simplified matched design.

X is genuinely discrete in this DGP; exact, disjoint pairs have identical X.
Under the stated no-hidden-confounding model, independent outcomes and a constant
mean effect justify the large-sample paired SE used here. Nearest-neighbor PSM,
reused controls and arbitrary real data do not inherit that guarantee.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))
from causal_utils import (exact_pairs, paired_rr_interval, evalue_interval,
                          rosenbaum_upper_p, method_options, matched_smd)

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05
rng = np.random.default_rng(6410)
n = 4000
activity = rng.integers(0, 9, n)
income = rng.choice([-1, 0, 1], n, p=[.25, .5, .25])
mega = rng.binomial(1, .3, n)
X = np.column_stack([activity, income, mega])
e = 1 / (1 + np.exp(-(-2.2 + .45*activity + .8*income + .7*mega)))
D = rng.binomial(1, e)
Y = 3 + .6*activity + .9*income + .7*mega + D + rng.normal(0, 1, n)
Y_pre = 3 + .55*activity + .85*income + .6*mega + rng.normal(0, 1, n)
# Artificial negative control: its lack of a D effect is specified, not inferred.
placebo_outcome = rng.poisson(2.5*np.exp(.35*income))
repeat = rng.binomial(1, 1/(1+np.exp(-(-1.6+.15*activity+.2*income+.6*D))))
idx_t, idx_c = exact_pairs(X, D, seed=0)


def paired_summary(outcome, it=idx_t, ic=idx_c):
    difference = outcome[it] - outcome[ic]
    estimate = difference.mean()
    se = difference.std(ddof=1) / np.sqrt(len(difference))
    critical = stats.t.ppf(1-ALPHA/2, len(difference)-1)
    return estimate, se, (estimate-critical*se, estimate+critical*se)


print("Точный матчинг на дискретных X; ATT среди оставленных treated")
print(f"Пар {len(idx_t)} из {D.sum()} treated; потеря {1-len(idx_t)/D.sum():.1%}")
print("SMD (фиксированная стандартизация до матчинга):",
      [round(matched_smd(X[:, j], D, idx_t, idx_c), 4) for j in range(3)])
results = [paired_summary(y) for y in [Y, placebo_outcome, Y_pre]]
for label, (estimate, se, ci) in zip(["Y", "placebo outcome", "placebo time"], results):
    print(f"{label}: оценка {estimate:+.3f}, SE {se:.3f}, CI [{ci[0]:+.3f}; {ci[1]:+.3f}]")
print("Интервал, включающий 0, не доказывает отсутствие важного placebo-сдвига.")

# A diagnostic on changed data, not the sampling distribution of original ATT.
placebo_estimates = []
for s in range(100):
    placebo_d = rng.permutation(D)
    pt, pc = exact_pairs(X, placebo_d, seed=s)
    placebo_estimates.append((Y[pt]-Y[pc]).mean())
placebo_estimates = np.asarray(placebo_estimates)
print("Placebo-D: медиана и 2.5/97.5 перцентили =",
      np.round(np.quantile(placebo_estimates, [.5, .025, .975]), 3))
print("Эти перцентили не используются как p-value основного ATT.")

rr, rr_lo, rr_hi = paired_rr_interval(repeat[idx_t], repeat[idx_c])
e_point, e_limit = evalue_interval(rr, rr_lo, rr_hi)
print(f"Парный RR={rr:.3f}, large-sample CI [{rr_lo:.3f}; {rr_hi:.3f}]")
print(f"E-value точки={e_point:.3f}, ближайшей границы={e_limit:.3f}")
print("E — равная сила двух связей; возможны неравные a,b с ab/(a+b−1) >= RR.")

# Prespecified positive alternative; sharp null and Rosenbaum odds-bound model.
difference = Y[idx_t]-Y[idx_c]
n_pairs, wins = (difference != 0).sum(), (difference > 0).sum()
gammas = np.arange(1., 5.001, .025)
pvalues = np.array([rosenbaum_upper_p(wins, n_pairs, g) for g in gammas])
first_cross = np.flatnonzero(pvalues >= ALPHA)
gamma_star = gammas[first_cross[0]] if len(first_cross) else None
label_gamma = f"первое пересечение на сетке: {gamma_star:.3f}" if gamma_star else ">5 (на сетке не найдено)"
print(f"Розенбаум, положительная альтернатива: {wins}/{n_pairs} побед; {label_gamma}")
print("У Γ нет общего порога «как A/B»: нужна предметная интерпретация скрытого отбора.")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
for j, (estimate, _, ci) in enumerate(results):
    axes[0].errorbar(j, estimate, yerr=[[estimate-ci[0]], [ci[1]-estimate]], fmt='o', capsize=4)
q = np.quantile(placebo_estimates, [.025, .5, .975])
axes[0].errorbar(3, q[1], yerr=[[q[1]-q[0]], [q[2]-q[1]]], fmt='s', capsize=4)
axes[0].axhline(0, color='grey'); axes[0].axhline(1, color='grey', ls=':')
axes[0].set_xticks(range(4), ['целевая Y', 'placebo outcome', 'placebo time', 'placebo D\nперцентили'], fontsize=8)
axes[0].set_title('Диагностика: интервалы оценок; отдельно — разброс placebo D', fontsize=10)
axes[1].plot(gammas, pvalues)
axes[1].axhline(ALPHA, color='red', ls='--', label='α=0,05')
if gamma_star is not None:
    axes[1].axvline(gamma_star, color='grey', ls=':', label=f'Γ*≈{gamma_star:.2f}')
axes[1].set(xlabel='Γ: допустимое отношение шансов назначения', ylabel='верхняя граница одностороннего p',
            title='Больше допустимого смещения — не меньше p-upper')
axes[1].legend(); fig.tight_layout()
fig.savefig(IMAGE_DIR/'practice_6_5_placebo_sensitivity.png', dpi=150); plt.close(fig)


def method_tree(verbose=True, **conditions):
    """Return candidate designs; subject-matter assumptions must be assessed first."""
    options = method_options(**conditions)
    if verbose:
        print('\n'.join(options))
    return options


scenarios = [
    ('Новая кнопка', dict(can_randomize=True)),
    ('Цена изменилась в части регионов; другие не затронуты',
     dict(already_happened=True, has_control_group=True, has_preperiod=True, parallel_trends=True)),
    ('Купон по непрерывному скорингу', dict(has_threshold=True)),
    ('Город-пилот с историей и незатронутыми донорами',
     dict(single_unit=True, has_preperiod=True, has_donors=True)),
    ('Подписка по желанию', dict(has_control_group=True, rich_covariates=True)),
    ('Рандомизированный encouragement, обоснованы IV-допущения', dict(has_instrument=True)),
    ('Единственный город без доноров/контрольных рядов', dict(single_unit=True, has_preperiod=True)),
]
for label, conditions in scenarios:
    print('\nСценарий:', label)
    method_tree(**conditions)

print("\nВопросы для отчёта:")
for question in [
    'Для каких treated оценён эффект и кого исключил матчинг?',
    'Почему X допустимы и достаточны; что известно о скрытом отборе?',
    'Соответствует ли SE дизайну, нет ли reused controls или cluster dependence?',
    'Какие существенные placebo-сдвиги исключены интервалами, а какие ещё допустимы?',
    'Какой скрытый отбор правдоподобен предметно, а не только в численной сетке?',
    'Связан ли outcome с бизнес-решением и куда допустимо переносить оценку?',
]:
    print('-', question)
