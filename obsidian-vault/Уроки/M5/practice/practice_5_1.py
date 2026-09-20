# -*- coding: utf-8 -*-
"""Практика 5.1 — Байесовский вывод: априор x правдоподобие -> апостериор (кейс «ЕдаДома»).

Принцип курса: не верь формуле — проверь симуляцией.

Что делаем:
1) монета с истинным p = 0.7: последовательное обновление трёх априоров
   Beta(1,1), Beta(5,2), Beta(50,50) по сетке 1/5/20/100/500 бросков —
   кривые апостериоров, сходимость, влияние априора тает (но медленно у сильного);
2) таблица «априор -> апостериор после 100 бросков»: среднее, 95% credible,
   P(p>0.5), прогноз следующего орла;
3) монетный двор (empirical Bayes): 1000 монет, честность каждой из Beta(6,4),
   20 тестовых бросков на монету — усохшие (shrinkage) оценки восстанавливают
   распределение честностей и бьют сырой MLE по MSE.

Запуск из корня репозитория: python3 course/modules/M5/practice/practice_5_1.py
Графики сохраняются в папку images/ этого модуля.
"""

# %% [markdown]
# Шаг 0. Импорты. Монета с p = 0.7 — «новый ресторан с конверсией 70%» в масштабе
# учебной монеты; модель та же Beta-Бернулли, что и для любой доли в продукте.

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260912)
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

P_TRUE = 0.7
N_FLIPS = 500
FLIPS = RNG.random(N_FLIPS) < P_TRUE  # True = орёл

PRIORS = {
    "Beta(1,1) плоский": (1, 1),
    "Beta(5,2) слабоинформативный": (5, 2),
    "Beta(50,50) сильный": (50, 50),
}
CUM_N = (1, 5, 20, 100, 500)
GRID = np.linspace(0.0, 1.0, 1001)


def posterior(a, b, heads, tails):
    """Апостериор Beta-Бернулли: апостериор = априор + данные."""
    return a + heads, b + tails


# %% [markdown]
# Шаг 1. Последовательное обновление: после каждого блока бросков вчерашний
# апостериор становится сегодняшним априором. Рисуем кривые апостериоров.

# %%
heads_cum = {n: int(FLIPS[:n].sum()) for n in CUM_N}
print("=" * 78)
print("1) ПОСЛЕДОВАТЕЛЬНОЕ ОБНОВЛЕНИЕ: монета с истинным p = 0.7")
print("-" * 78)
for n in CUM_N:
    h = heads_cum[n]
    print(f"   после {n:>3} бросков: орлов {h:>3} ({h / n:.1%}), решек {n - h:>3}")
print()

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
cmap = plt.get_cmap("viridis")
for ax, (name, (a, b)) in zip(axes, PRIORS.items()):
    ax.plot(GRID, stats.beta.pdf(GRID, a, b), color="gray", ls="--", lw=1.6,
            label=f"априор {name}\n(вес {a + b} псевдобросков)")
    for i, n in enumerate(CUM_N):
        h = heads_cum[n]
        pa, pb = posterior(a, b, h, n - h)
        ax.plot(GRID, stats.beta.pdf(GRID, pa, pb), color=cmap(i / (len(CUM_N) - 1)),
                lw=2.2, label=f"n={n}: Beta({pa},{pb}), ср. {pa / (pa + pb):.3f}")
    ax.axvline(P_TRUE, color="#C44E52", lw=2.5, alpha=0.8)
    ax.text(P_TRUE + 0.02, ax.get_ylim()[1] * 0.92, "истина 0.7", color="#C44E52",
            fontsize=9)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("p")
    ax.legend(fontsize=6.6, loc="upper left")
axes[0].set_ylabel("плотность апостериора")
fig.suptitle("Данные заливают априор: при 500 бросках апостериоры почти совпали, "
             "но Beta(50,50) ещё тянет влево", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_5_1_sequential.png", dpi=150)
plt.close(fig)
print("Сохранено: practice_5_1_sequential.png")

# %% [markdown]
# Шаг 2. Таблица «априор -> апостериор после 100 бросков» + прогноз следующего орла
# (апостериорное прогнозное = среднее апостериора).

# %%
N_TABLE = 100
h100, t100 = heads_cum[N_TABLE], N_TABLE - heads_cum[N_TABLE]
rows = []
for name, (a, b) in PRIORS.items():
    pa, pb = posterior(a, b, h100, t100)
    lo, hi = stats.beta.ppf([0.025, 0.975], pa, pb)
    p_gt_half = 1 - stats.beta.cdf(0.5, pa, pb)
    p_half_txt = f"{p_gt_half:.2%}" if p_gt_half > 0.99 else f"{p_gt_half:.1%}"
    rows.append({
        "априор": name,
        "вес априора": a + b,
        "ср. априора": f"{a / (a + b):.2f}",
        "апостериор": f"Beta({pa},{pb})",
        "ср. апостериора": f"{pa / (pa + pb):.3f}",
        "95% credible": f"[{lo:.3f}; {hi:.3f}]",
        "P(p>0.5|D)": p_half_txt,
        "P(орёл|D)": f"{pa / (pa + pb):.1%}",
    })
df = pd.DataFrame(rows)
print("=" * 78)
print(f"2) АПРИОР -> АПОСТЕРИОР ПОСЛЕ {N_TABLE} БРОСКОВ ({h100} орлов / {t100} решек)")
print("-" * 78)
print(df.to_string(index=False))

means = {name: (a + h100) / (a + b + N_TABLE) for name, (a, b) in PRIORS.items()}
spread_100 = max(means.values()) - min(means.values())
h500, t500 = heads_cum[500], 500 - heads_cum[500]
means_500 = {name: (a + h500) / (a + b + 500) for name, (a, b) in PRIORS.items()}
spread_500 = max(means_500.values()) - min(means_500.values())
print(f"\n   разброс средних между априорами: после 100 бросков {spread_100:.3f},"
      f" после 500 — {spread_500:.3f} (влияние априора тает как 1/n)")
for name in PRIORS:
    a, b = PRIORS[name]
    print(f"   {name:<32}: ср. при n=100 -> {means[name]:.3f}, при n=500 -> {means_500[name]:.3f}")

# %%
fig, ax = plt.subplots(figsize=(9.6, 4.6))
colors = {"Beta(1,1) плоский": "#4C72B0", "Beta(5,2) слабоинформативный": "#55A868",
          "Beta(50,50) сильный": "#DD8452"}
for name, (a, b) in PRIORS.items():
    ax.plot(GRID, stats.beta.pdf(GRID, a, b), color=colors[name], ls=":", lw=1.4,
            alpha=0.7, label=f"{name} (априор)")
    pa, pb = posterior(a, b, h100, t100)
    ax.plot(GRID, stats.beta.pdf(GRID, pa, pb), color=colors[name], lw=1.8, alpha=0.75,
            label=f"апостериор при n=100 (ср. {pa / (pa + pb):.3f})")
    pa5, pb5 = posterior(a, b, h500, t500)
    ax.plot(GRID, stats.beta.pdf(GRID, pa5, pb5), color=colors[name], lw=3.0,
            label=f"апостериор при n=500 (ср. {pa5 / (pa5 + pb5):.3f})")
ax.axvline(P_TRUE, color="#C44E52", lw=2.5, alpha=0.9)
ax.text(P_TRUE + 0.015, 12.2, "истина 0.7", color="#C44E52", fontsize=10)
ax.set_xlabel("p")
ax.set_ylabel("плотность")
ax.set_title("Чувствительность к априору: три разных убеждения, 100 и 500 бросков данных")
ax.legend(fontsize=7.6, loc="upper left")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_5_1_priors_fade.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_5_1_priors_fade.png")

# %% [markdown]
# Шаг 3. Empirical Bayes: монетный двор. Честности 1000 монет ~ Beta(6,4)
# (= априор, восстановленный по истории монет двора). Каждую монету бросили
# всего 5 раз (как у ресторана в первый день: пара показов карточки).
# Сравниваем сырой MLE (орлы/5) и байесовскую оценку.

# %%
A0, B0 = 6, 4          # априор Beta(6,4): средняя честность 0.6, вес 10 бросков
N_COINS, N_TEST = 1000, 5
p_true = RNG.beta(A0, B0, N_COINS)
k = RNG.binomial(N_TEST, p_true)
mle = k / N_TEST
# Исторические объекты не совпадают с оцениваемыми монетами.
from scipy.optimize import minimize
n_history = np.full(1000, 20)
k_history = RNG.binomial(n_history, RNG.beta(A0, B0, len(n_history)))
fit = minimize(lambda log_ab: -stats.betabinom.logpmf(
    k_history, n_history, *np.exp(log_ab)).sum(), np.log([2., 2.]), method="L-BFGS-B",
    bounds=[(-8, 10), (-8, 10)])
if not fit.success:
    raise RuntimeError(f"Beta-binomial fit не сошёлся: {fit.message}")
a_est, b_est = np.exp(fit.x)
post_mean = (a_est + k) / (a_est + b_est + N_TEST)
oracle_mean = (A0 + k) / (A0 + B0 + N_TEST)
print(f"Estimated historical prior Beta({a_est:.2f}, {b_est:.2f}); "
      f"oracle MSE={np.mean((oracle_mean-p_true)**2):.4f}")

mse_mle = float(np.mean((mle - p_true) ** 2))
mse_post = float(np.mean((post_mean - p_true) ** 2))
extreme = float(np.mean((mle <= 0.0) | (mle >= 1.0)))

print("=" * 78)
print(f"3) MONETНЫЙ ДВОР (empirical Bayes): 1000 монет, честность ~ Beta(6,4),")
print(f"   по {N_TEST} бросков на монету")
print("-" * 78)
print(f"   сырой MLE          : MSE = {mse_mle:.4f}, доля крайних (0% или 100%) = {extreme:.1%}")
print(f"   байесовская оценка : MSE = {mse_post:.4f}  (в {mse_mle / mse_post:.2f} раза точнее)")
print(f"   средняя честность  : истина {p_true.mean():.3f}, MLE {mle.mean():.3f}, "
      f"байес {post_mean.mean():.3f}")
print(f"   SD разброса оценок : истина {p_true.std():.3f}, MLE {mle.std():.3f}, "
      f"байес {post_mean.std():.3f} (MLE шумит сильнее истины)")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
axes[0].hist(p_true, bins=30, range=(0, 1), color="#8172B3", alpha=0.55,
             label=f"истинные честности (Beta({A0},{B0}))")
axes[0].hist(mle, bins=30, range=(0, 1), color="#C44E52", alpha=0.55,
             histtype="stepfilled", label=f"сырой MLE по {N_TEST} броскам")
axes[0].hist(post_mean, bins=30, range=(0, 1), color="#4C72B0", alpha=0.55,
             histtype="step", lw=2.0, label="байесовские оценки")
axes[0].set_xlabel("честность монеты p")
axes[0].set_ylabel("число монет")
axes[0].set_title(f"Posterior mean уменьшает ошибку: {extreme:.0%} монет с MLE 0%/100%\n"
                  f"усохли к среднему двора — «монет с честностью 0 не бывает»")
axes[0].legend(fontsize=8)

axes[1].scatter(mle, post_mean, s=9, color="#4C72B0", alpha=0.45)
lims = [-0.03, 1.03]
axes[1].plot(lims, lims, color="gray", ls="--", lw=1.2, label="y = x (оценка = MLE)")
axes[1].axhline(A0 / (A0 + B0), color="#55A868", lw=1.6, ls=":",
                label=f"средняя двора {A0 / (A0 + B0):.1f}")
axes[1].set_xlim(lims)
axes[1].set_ylim(lims)
axes[1].set_xlabel(f"сырой MLE по {N_TEST} броскам")
axes[1].set_ylabel("апостериорное среднее")
axes[1].set_title(f"Shrinkage: крайние оценки тянутся к среднему двора,\n"
                  f"MSE {mse_mle:.4f} -> {mse_post:.4f} (x{mse_mle / mse_post:.2f} точнее)")
axes[1].legend(fontsize=8)
fig.suptitle("Empirical Bayes: априор по отдельной истории + 5 бросков на новую монету",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_5_1_empirical_bayes.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_5_1_empirical_bayes.png")

# %%
print()
print("=" * 78)
print("ГЛАВНОЕ (что должны увидеть):")
print("1) Апостериор = априор + данные: орлы идут в alpha, решки в beta. Все три")
print("   априора дают апостериоры, стягивающиеся к истине 0.7 по мере бросков.")
print("2) Влияние априора тает как 1/n, но у Beta(50,50) вес = 100 псевдобросков:")
print("   и после 100 бросков он ещё тянет среднее вниз (вдвое меньше данных —")
print("   вдвое сильнее тянет). Плоский априор = минимальное вмешательство.")
print("3) Прогноз следующего орла = среднее апостериора: никакой магии, то же число.")
print("4) Empirical Bayes: априор оценён по независимой истории и сглаживает крайние")
print("   MLE по 5 броскам (0% и 100%!) обратно в разумный диапазон и сильно")
print("   выигрывает по MSE — так скорят новые рестораны «ЕдаДома» с малым трафиком.")
