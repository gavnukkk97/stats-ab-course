# -*- coding: utf-8 -*-
"""Практика 2.1 — Логика NHST: p-value, ошибки I/II рода, мощность (кейс «ЕдаДома»).

Принцип курса: не верь формуле — проверь симуляцией.

Что делаем:
1) (а) 10 000 A/A-тестов при верной H0: гистограмма p-value равномерна,
      доля p<0.05 примерно равна α = 5%;
2) (б) A/B с реальным эффектом: p-value скошен к нулю,
      доля p<0.05 = мощность теста;
3) (в) кривые мощности от n для трёх эффектов: симуляция против теории;
4) (г) Type M: при малой мощности «значимые» оценки эффекта преувеличены
      в разы; заодно Type S — доля значимых эффектов не в ту сторону.

Запуск:  python3 practice_2_1.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# Шаг 0. Импорты и модель данных.
# Выручка на юзера за неделю ~ N(1240, 300^2) — учебная модель «ЕдаДома».
# (Поведение теста на скошенном логнормальном чеке — следующий урок, 2.2.)

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RNG = np.random.default_rng(20260908)
HERE = Path(__file__).resolve().parent

MU, SIGMA = 1240.0, 300.0  # средняя выручка на юзера, руб., и её SD
ALPHA = 0.05


def ab_pvalues(effect, n_per_group, reps):
    """`reps` виртуальных A/B-тестов -> массив p-value t-теста Уэлча.

    equal_var=False не требует равенства дисперсий — почему именно Уэлч,
    разбираем в уроке 2.2; здесь это просто «честный t-тест».
    """
    a = RNG.normal(MU, SIGMA, size=(reps, n_per_group))
    b = RNG.normal(MU + effect, SIGMA, size=(reps, n_per_group))
    return stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue


def power_theory(effect, n):
    """Теоретическая мощность двустороннего z-теста: Phi(d*sqrt(n/2) - 1.96)."""
    z_a = stats.norm.ppf(1 - ALPHA / 2)
    return stats.norm.cdf(effect / SIGMA * np.sqrt(np.asarray(n, dtype=float) / 2.0) - z_a)


# %% [markdown]
# Шаг 1. (а) A/A-тесты: эффекта нет, H0 верна. (б) A/B-тесты: эффект +60 руб.
# Смотрим на распределение 10 000 p-value в обоих мирах.

# %%
N_AA, N_AB, N_PER = 10_000, 10_000, 200
EFFECT_AB = 60.0

p_aa = ab_pvalues(0.0, N_PER, N_AA)
p_ab = ab_pvalues(EFFECT_AB, N_PER, N_AB)

ks = stats.kstest(p_aa, "uniform")
quart = [
    np.mean((p_aa >= lo) & (p_aa < hi))
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))
]

print("=" * 78)
print("1) РАСПРЕДЕЛЕНИЕ p-value: n = %d на группу, по %d тестов" % (N_PER, N_AA))
print("-" * 78)
print("   (а) A/A — эффекта нет (H0 верна):")
print(f"       средний p-value      = {p_aa.mean():.3f}   (у равномерного на [0;1]: 0.500)")
print(f"       доля p<0.05          = {np.mean(p_aa < 0.05):.2%}  (номинал alpha = 5%)")
print(f"       доля p<0.01          = {np.mean(p_aa < 0.01):.2%}  (номинал 1%)")
print(f"       доли по квартилам    = "
      f"{quart[0]:.2f} / {quart[1]:.2f} / {quart[2]:.2f} / {quart[3]:.2f} (у равномерного — по 0.25)")
print(f"       KS-тест равномерности: p = {ks.pvalue:.2f}  (равномерность не отвергается)")
print()
print("   (б) A/B — реальный эффект +60 руб. (d = 60/300 = 0.2, H1 верна):")
print(f"       доля p<0.05 = МОЩНОСТЬ = {np.mean(p_ab < 0.05):.1%}  "
      f"(теория: {power_theory(EFFECT_AB, N_PER):.1%})")
print(f"       медианный p-value    = {np.median(p_ab):.3f}   (в A/A был ~0.5)")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
axes[0].hist(p_aa, bins=40, color="#4C72B0", alpha=0.85)
axes[0].axhline(N_AA / 40, color="#C44E52", ls="--", lw=1.5, label="ожидание при равномерности")
axes[0].axvline(ALPHA, color="k", lw=1.5)
axes[0].text(0.96, 0.88, f"доля p<0.05 = {np.mean(p_aa < 0.05):.1%} ≈ alpha",
             transform=axes[0].transAxes, ha="right", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
axes[0].set_title("(а) A/A: эффекта нет — p-value равномерен")
axes[0].set_xlabel("p-value")
axes[0].set_ylabel("число тестов")
axes[0].legend(fontsize=8)

axes[1].hist(p_ab, bins=40, color="#DD8452", alpha=0.9)
axes[1].axvline(ALPHA, color="k", lw=1.5)
axes[1].text(0.96, 0.88, f"доля p<0.05 = мощность {np.mean(p_ab < 0.05):.0%}",
             transform=axes[1].transAxes, ha="right", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
axes[1].set_title("(б) A/B: эффект +60 руб. — p сыпется к нулю")
axes[1].set_xlabel("p-value")

fig.suptitle("10 000 виртуальных тестов: p-value — портрет критерия",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "practice_2_1_pvalue_hist.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_2_1_pvalue_hist.png")

# %% [markdown]
# Шаг 2. (в) Кривые мощности от n для трёх эффектов (+30 / +60 / +120 руб.).
# Точки — симуляция (1000 тестов на точку), линии — нормальная теория.

# %%
ns = np.arange(100, 3001, 100)
effects = (30.0, 60.0, 120.0)
REPS_CURVE = 1_000
colors = {30.0: "#C44E52", 60.0: "#4C72B0", 120.0: "#55A868"}

print("=" * 78)
print("2) МОЩНОСТЬ ОТ n: три эффекта (alpha = 5%, двусторонний тест)")
print("-" * 78)
print(f"   {'эффект':>9} | {'d':>5} | {'n для 80% теории':>16} | {'n для 80% симуляции':>19}")
power_sim = {}
for e in effects:
    pw = np.array([np.mean(ab_pvalues(e, int(n), REPS_CURVE) < ALPHA) for n in ns])
    power_sim[e] = pw
    n80_th = 2 * (stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(0.8)) ** 2 * (SIGMA / e) ** 2
    n80_sim = float(np.interp(0.8, pw, ns))
    print(f"   {e:>7.0f} руб | {e / SIGMA:>5.2f} | {n80_th:>16.0f} | {n80_sim:>19.0f}")

# %%
fig, ax = plt.subplots(figsize=(8.6, 4.5))
for e in effects:
    ax.plot(ns, power_sim[e], "o", ms=3.5, color=colors[e], alpha=0.55)
    ax.plot(ns, power_theory(e, ns), "-", lw=2.2, color=colors[e],
            label=f"эффект +{e:.0f} руб. (d = {e / SIGMA:.2f})")
ax.axhline(0.8, color="gray", ls=":", lw=1.5)
ax.text(2450, 0.82, "power 80%", fontsize=9)
ax.axhline(ALPHA, color="gray", ls=":", lw=1.5)
ax.text(2450, 0.07, "alpha = 5%", fontsize=9)
ax.set_xlabel("n на группу")
ax.set_ylabel("мощность = доля p<0.05")
ax.set_title("Мощность растёт с n и с величиной эффекта:\nточки — симуляция, линии — теория")
ax.legend(fontsize=9, loc="center right")
fig.tight_layout()
fig.savefig(HERE / "practice_2_1_power_curves.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_2_1_power_curves.png")

# %% [markdown]
# Шаг 3. (г) Type M и Type S: истинный эффект +30 руб. при n = 250 на группу —
# мощность всего ~20%. Смотрим на оценки эффекта среди «значимых» тестов.

# %%
TRUE_EFF, N_LOW, REPS_TM = 30.0, 250, 30_000
a = RNG.normal(MU, SIGMA, size=(REPS_TM, N_LOW))
b = RNG.normal(MU + TRUE_EFF, SIGMA, size=(REPS_TM, N_LOW))
pv = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue
eff = b.mean(axis=1) - a.mean(axis=1)
sig = pv < ALPHA
eff_sig = eff[sig]

print("=" * 78)
print("3) Type M / Type S: истинный эффект +30 руб., n = 250 на группу")
print("-" * 78)
print(f"   эмпирическая мощность                : {sig.mean():.1%} "
      f"(теория {power_theory(TRUE_EFF, N_LOW):.0%})")
print(f"   средняя оценка, все {REPS_TM} тестов  : {eff.mean():+.1f} руб. (несмещённо: +30)")
print(f"   средняя оценка среди значимых        : {eff_sig.mean():+.1f} руб.")
print(f"   Type M (преувеличение)               : x{eff_sig.mean() / TRUE_EFF:.2f}")
print(f"   Type S (значимые не в ту сторону)    : {np.mean(eff_sig < 0):.2%} от значимых")

# %%
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.hist(eff, bins=80, color="#bbbbbb", alpha=0.55, label=f"все {REPS_TM} тестов")
ax.hist(eff_sig, bins=80, color="#4C72B0", alpha=0.9,
        label=f"только значимые, p<0.05 ({sig.mean():.0%} тестов)")
ax.axvline(TRUE_EFF, color="#55A868", lw=3, label=f"истинный эффект +{TRUE_EFF:.0f} руб.")
ax.axvline(eff_sig.mean(), color="#C44E52", lw=2.2, ls="--",
           label=f"среднее значимых {eff_sig.mean():+.0f} руб. (x{eff_sig.mean() / TRUE_EFF:.1f})")
ax.set_xlabel("оценка эффекта, руб.")
ax.set_ylabel("число тестов")
ax.set_title("Type M: порог значимости отбирает «удачные» выборки —\n"
             "значимая оценка в разы больше истинного эффекта")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "practice_2_1_type_m.png", dpi=150)
plt.close(fig)
print("Сохранено: practice_2_1_type_m.png")

# %%
print()
print("=" * 78)
print("ГЛАВНОЕ (что должны увидеть):")
print("1) Под H0 p-value РАВНОМЕРЕН на [0;1]: доли p<0.05 и p<0.01 совпадают с")
print("   номиналами, квартилы по 25%. Каждый 20-й A/A-тест «прокрасится» — это")
print("   свойство критерия, а не поломка сплит-системы (основа A/A-аудита, М4).")
print("2) Под H1 p-value скошен к нулю; доля p<0.05 = мощности теста.")
print("3) Мощность держится на трёх рычагах: n, величина эффекта, alpha.")
print("   Для 80% мощности: +30 руб. (d=0.1) ~ 1570 юзеров на группу,")
print("   +60 руб. (d=0.2) ~ 393, +120 руб. (d=0.4) ~ 99.")
print("4) При мощности ~20% значимые оценки в среднем примерно вдвое больше")
print("   истины (Type M) и изредка не в ту сторону (Type S). «Прокрас» слабого")
print("   теста — лотерея с завышенным выигрышем, а не измерение эффекта.")
