# -*- coding: utf-8 -*-
"""Практика 1.3 — ЦПТ и доверительные интервалы (сквозной кейс «ЕдаДома»).

Принцип курса: не верь формуле — проверь симуляцией.

Что делаем:
1) 5000 раз сэмплируем по n = {5, 30, 100, 1000} наблюдений из двух «генеральных
   совокупностей»: скошенный логнормальный чек заказа и нескошенное равномерное
   время доставки; строим распределение СРЕДНИХ (гистограммы 2x4);
2) показываем сужение стандартной ошибки ~ 1/sqrt(n) на log-log графике;
3) считаем эмпирическое покрытие 95% t-интервала при разных n.

Запуск:  python3 practice_1_3.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# Шаг 0. Импорты и «генеральная совокупность» ЕдаДома.
# Нам как симулятору известны истинные параметры — сравниваем с ними оценки.

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RNG = np.random.default_rng(20260907)  # seed -> результат воспроизводим
HERE = Path(__file__).resolve().parent

N_SIM = 5_000          # сколько раз «повторяем эксперимент»
SAMPLE_SIZES = (5, 30, 100, 1000)
CONF = 0.95

# Чек заказа: логнормал (медиана 700 руб., сигма логарифмов 1.2) — типичная
# форма продуктовых денежных метрик: правая асимметрия, тяжёлый хвост.
MEDIAN_CHECK = 700.0
SIGMA_LOG = 1.2
TRUE_MEAN_CHECK = MEDIAN_CHECK * np.exp(SIGMA_LOG**2 / 2)
TRUE_SD_CHECK = MEDIAN_CHECK * np.sqrt((np.exp(SIGMA_LOG**2) - 1) * np.exp(SIGMA_LOG**2))
SKEW_CHECK = (np.exp(SIGMA_LOG**2) + 2) * np.sqrt(np.exp(SIGMA_LOG**2) - 1)

# Время доставки: равномерное на [20, 60] минут — симметрично, без хвостов.
DELIVERY_A, DELIVERY_B = 20.0, 60.0
TRUE_MEAN_DELIVERY = (DELIVERY_A + DELIVERY_B) / 2
TRUE_SD_DELIVERY = (DELIVERY_B - DELIVERY_A) / np.sqrt(12)

DISTS = {
    "check": {
        "label": "Чек, руб. (логнормал)",
        "true_mean": TRUE_MEAN_CHECK,
        "true_sd": TRUE_SD_CHECK,
        "skew": SKEW_CHECK,
    },
    "delivery": {
        "label": "Доставка, мин (равномерное)",
        "true_mean": TRUE_MEAN_DELIVERY,
        "true_sd": TRUE_SD_DELIVERY,
        "skew": 0.0,
    },
}


def draw(dist: str, size) -> np.ndarray:
    """Сэмплируем сырые наблюдения из выбранной совокупности."""
    if dist == "check":
        return MEDIAN_CHECK * RNG.lognormal(0.0, SIGMA_LOG, size=size)
    return RNG.uniform(DELIVERY_A, DELIVERY_B, size=size)


# %% [markdown]
# Шаг 1. Главная симуляция: 5000 реализаций x̄ при каждом n.
# Для каждой реализации строим t-интервал x̄ ± t·s/√n и проверяем,
# накрыл ли он истинное среднее. Доля накрытий = эмпирическое покрытие.

# %%
for name, d in DISTS.items():
    d["means"], d["emp_se"], d["theory_se"], d["coverage"] = {}, [], [], []
    for n in SAMPLE_SIZES:
        x = draw(name, size=(N_SIM, n))                  # сырые выборки
        means = x.mean(axis=1)                           # 5000 реализаций x̄
        sds = x.std(axis=1, ddof=1)                      # 5000 оценок s
        d["means"][n] = means
        d["emp_se"].append(means.std(ddof=1))            # эмпирическая SE
        d["theory_se"].append(d["true_sd"] / np.sqrt(n))  # теория: σ/√n
        t_crit = stats.t.ppf(0.5 + CONF / 2, df=n - 1)
        half = t_crit * sds / np.sqrt(n)
        covered = (d["true_mean"] >= means - half) & (d["true_mean"] <= means + half)
        d["coverage"].append(covered.mean())

print("=" * 80)
print("1) Стандартная ошибка среднего: теория (σ/√n) против симуляции")
print("   и эмпирическое покрытие 95% t-интервала")
for name, d in DISTS.items():
    print(f"\n   {d['label']}; истинное среднее μ = {d['true_mean']:.1f}")
    print(f"   {'n':>6} | {'SE теор':>10} | {'SE эмп':>10} | {'покрытие t-CI':>13}")
    for n, ts, es, cov in zip(SAMPLE_SIZES, d["theory_se"], d["emp_se"], d["coverage"]):
        print(f"   {n:>6} | {ts:>10.2f} | {es:>10.2f} | {cov:>12.1%}")

print()
print("=" * 80)
print("2) Скорость сходимости ЦПТ: правило Денга ~100·s² наблюдений")
print(f"   Чек: асимметрия s = {SKEW_CHECK:.1f} (считаем по формуле логнормала)")
print(f"   -> для нормальной аппроксимации средних нужно порядка "
      f"{100 * SKEW_CHECK**2:,.0f} наблюдений")
print("   (варианты: n > 122.56·s² — Денг; n > 355·s² — Kohavi et al.)")
print(f"   При n = 1000 < {100 * SKEW_CHECK**2:,.0f} сходимость ЕЩЁ не наступила:")
print(f"   покрытие t-CI для чека ниже номинальных 95%.")
print(f"   Доставка (асимметрия 0): t-CI держит 95% уже при малых n.")

# %% [markdown]
# Шаг 2. Гистограммы 2×4: как распределение СРЕДНИХ (не данных!)
# становится нормальным с ростом n — и как логнормал тормозит этот процесс.

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for row, (name, d) in enumerate(DISTS.items()):
    for col, n in enumerate(SAMPLE_SIZES):
        ax = axes[row, col]
        means = d["means"][n]
        ax.hist(means, bins=60, density=True, color="#4C72B0", alpha=0.75)
        grid = np.linspace(means.min(), np.quantile(means, 0.999), 400)
        ax.plot(
            grid,
            stats.norm.pdf(grid, d["true_mean"], d["true_sd"] / np.sqrt(n)),
            color="#C44E52", lw=2,
            label="ЦПТ: N(μ, σ²/n)",
        )
        ax.axvline(d["true_mean"], color="k", ls="--", lw=1, label="истинное μ")
        if name == "check":  # редкий огромный средний расплющивает картинку — подрежем хвост
            ax.set_xlim(means.min(), np.quantile(means, 0.99))
        ax.set_title(f"{d['label']}, n={n}", fontsize=10)
        if row == 1:
            ax.set_xlabel("среднее по выборке")
        if col == 0:
            ax.set_ylabel("плотность")
        ax.legend(fontsize=7)

fig.suptitle("ЦПТ в действии: 5000 средних при разных n (кейс «ЕдаДома»)", fontsize=13)
fig.tight_layout()
fig.savefig(HERE / "clt_means_histograms.png", dpi=150)
print("\nСохранено: clt_means_histograms.png")

# %% [markdown]
# Шаг 3. Сужение SE ~ 1/√n: log-log график. Прямая с наклоном −0.5 —
# «закон убывающей отдачи»: вдвое точнее = вчетверо больше данных.

# %%
fig, ax = plt.subplots(figsize=(7.5, 5))
colors = {"check": "#4C72B0", "delivery": "#55A868"}
print("=" * 80)
print("3) Сужение SE с ростом n (наклон в координатах log-log)")
for name, d in DISTS.items():
    ax.loglog(SAMPLE_SIZES, d["emp_se"], "o-", color=colors[name],
              label=f"{d['label']}: эмпирика")
    ax.loglog(SAMPLE_SIZES, d["theory_se"], "s--", color=colors[name], alpha=0.45,
              label=f"{d['label']}: σ/√n")
    slope = np.polyfit(np.log10(SAMPLE_SIZES), np.log10(d["emp_se"]), 1)[0]
    print(f"   {d['label']}: наклон = {slope:.3f} (теория: −0.500)")

ax.set_xlabel("Объём выборки n (log)")
ax.set_ylabel("SE среднего (log)")
ax.set_title("SE ~ 1/√n: вдвое уже — вчетверо больше данных")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "se_scaling_loglog.png", dpi=150)
print("Сохранено: se_scaling_loglog.png")

# %%
print()
print("=" * 80)
print("ГЛАВНОЕ (что должны увидеть):")
print(f"1) Эмпирическая SE совпадает с σ/√n при всех n и для обеих метрик.")
print(f"2) Наклон SE(n) ≈ −0.5: чтобы сузить интервал вдвое, нужно ×4 данных.")
skew_cov = dict(zip(SAMPLE_SIZES, DISTS["check"]["coverage"]))
print(f"3) Чек (s = {SKEW_CHECK:.1f}): при n=5 средние сильно скошены вправо, "
      f"покрытие {skew_cov[5]:.1%};")
print(f"   при n=1000 гистограмма выглядит нормально, но покрытие {skew_cov[1000]:.1%} —")
print("   «выглядит нормальным» ≠ «аппроксимация достаточна»; по Денгу нужно ~12500.")
print(f"4) Доставка (без асимметрии): покрытие t-CI у 95% при всех n от 30.")
