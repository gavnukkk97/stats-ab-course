# -*- coding: utf-8 -*-
"""Практика 3.1 — Дизайн АБ-теста: размер выборки, MDE, мощность, сплит, длительность
(кейс «ЕдаДома»).

Принцип курса: не верь формуле — проверь симуляцией.

Что делаем:
1) Калькулятор n <-> MDE <-> power (z-формулы) + сверка симуляцией:
   на расчётном n мощность ~80%, на «половинной» выборке (забытый
   множитель 2) — ~50.8% (сюжет поста shelter_analytics);
2) кривая n(MDE) с аннотацией ошибки x2 (8 289 -> 16 578; 8 711 -> 17 422);
3) мощность от сплита r при фиксированном трафике: 50/50 против перекосов;
4) сколько дней ждать: дневной трафик с выходными x1.4, полные недели.

Запуск из корня репозитория: python3 course/modules/M3/practice/practice_3_1.py
Графики сохраняются в папку images/ этого модуля.
"""

# %% [markdown]
# Шаг 0. Импорты и параметры «ЕдаДома»: конверсия в заказ p0 = 12%,
# метрика на юзера бинарная => sigma^2 = p0(1-p0) = 0.1056.

# %%
import math

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RNG = np.random.default_rng(20260908)
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

P0 = 0.12
SIGMA = math.sqrt(P0 * (1 - P0))  # 0.325
ALPHA = 0.05
Z_A = stats.norm.ppf(1 - ALPHA / 2)  # 1.96
Z_B = stats.norm.ppf(0.8)  # 0.84


# --- z-формулы дизайна (двухвыборочный мир) -------------------------------
def n_per_group(mde, sigma=SIGMA, alpha=ALPHA, power=0.8):
    """Правильная формула: n на группу для двухвыборочного теста."""
    z1 = stats.norm.ppf(1 - alpha / 2)
    z2 = stats.norm.ppf(power)
    return 2 * (z1 + z2) ** 2 * sigma**2 / mde**2


def n_one_sample(mde, sigma=SIGMA, alpha=ALPHA, power=0.8):
    """ОШИБОЧНАЯ формула (без x2) — одновыборочная, для контраста."""
    z1 = stats.norm.ppf(1 - alpha / 2)
    z2 = stats.norm.ppf(power)
    return (z1 + z2) ** 2 * sigma**2 / mde**2


def power_z(mde, n, sigma=SIGMA, alpha=ALPHA):
    """Мощность z-теста разности долей при n на группу (нормальная теория)."""
    se = sigma * math.sqrt(2.0 / n)
    critical = stats.norm.ppf(1 - alpha / 2)
    return stats.norm.cdf(mde / se - critical) + stats.norm.cdf(-mde / se - critical)


def mde_at_n(n, sigma=SIGMA, alpha=ALPHA, power=0.8):
    """MDE, который ловится с данной мощностью при n на группу."""
    z1 = stats.norm.ppf(1 - alpha / 2)
    z2 = stats.norm.ppf(power)
    return (z1 + z2) * sigma * math.sqrt(2.0 / n)


def simulate_power(mde, n, reps=3000, rng=None):
    """Доля p<0.05 среди `reps` виртуальных A/B (t-тест на 0/1 юзерах)."""
    rng = rng or RNG
    a = rng.binomial(n, P0, size=reps) / n
    b = rng.binomial(n, P0 + mde, size=reps) / n
    # Достаточные статистики бинарных данных: тот же Welch, без матриц reps×n.
    va, vb = a * (1-a) / (n-1), b * (1-b) / (n-1)
    se2 = va + vb
    df = se2**2 / ((va**2 + vb**2) / (n-1))
    p = 2 * stats.t.sf(np.abs(b-a) / np.sqrt(se2), df)
    return float(np.mean(p < ALPHA))


# %% [markdown]
# Шаг 1. Калькулятор n<->MDE<->power и сверка симуляцией.
# Ключевая проверка: расчётное n даёт ~80% мощности, а ПОЛОВИНА n
# (забытый множитель 2) — около 50.8% в нормальной аппроксимации.

# %%
print("=" * 78)
print("1) КАЛЬКУЛЯТОР n / MDE / POWER (p0 = 12%, alpha = 5%, power = 80%)")
print("-" * 78)
print(f"   {'MDE':>10} | {'n без x2':>9} | {'n верное':>9} | {'мощн.на верном':>14}"
      f" | {'мощн.на половине':>17}")
print(f"   {'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*14}-+-{'-'*17}")
for mde_pp in (2.0, 1.5, 1.0, 0.7):
    mde = mde_pp / 100
    n_wrong = math.ceil(n_one_sample(mde))
    n_right = math.ceil(n_per_group(mde))
    pw_right = simulate_power(mde, n_right)
    pw_half = simulate_power(mde, n_wrong)  # половина требуемого
    print(f"   {mde_pp:>7.1f}пп | {n_wrong:>9,} | {n_right:>9,} | "
          f"{pw_right:>13.1%} | {pw_half:>16.1%}")

half_theory = stats.norm.cdf((Z_A + Z_B) / math.sqrt(2) - Z_A)
print(f"\n   Теория для «половины выборки»: Phi((z_a+z_b)/sqrt(2) - z_a) = "
      f"{half_theory:.1%}  <- ровно столько в посте shelter_analytics")
print(f"   MDE при 5 000 на группу: {mde_at_n(5000)*100:.2f} п.п.;  "
      f"при 16 578: {mde_at_n(16578)*100:.2f} п.п.")

# %% [markdown]
# Шаг 2. Кривая n(MDE): наивная линия против правильной — зримая «вилка x2».

# %%
mdes = np.linspace(0.5, 3.0, 60) / 100
n_wrong_curve = np.array([n_one_sample(m) for m in mdes])
n_right_curve = 2 * n_wrong_curve

fig, ax = plt.subplots(figsize=(8.8, 4.6))
ax.plot(mdes * 100, n_wrong_curve, "--", lw=2, color="#C44E52",
        label="без x2 (одновыборочная формула) — ОШИБКА")
ax.plot(mdes * 100, n_right_curve, "-", lw=2.4, color="#4C72B0",
        label="правильно: n на группу (двухвыборочная)")
ax.set_yscale("log")
ax.set_xlabel("MDE, п.п. к конверсии 12%")
ax.set_ylabel("n на группу (log)")
ax.set_title("Цена MDE и главная ошибка дизайна: забытый множитель 2\n"
             "(тест теряет половину выборки и мощность падает до ~50.8%)")
# аннотация на MDE = 1 п.п.
x0, y_wrong, y_right = 1.0, n_one_sample(0.01), n_per_group(0.01)
ax.plot([x0, x0], [y_wrong, y_right], color="k", lw=1.6)
ax.annotate(f"{y_wrong:,.0f}", (x0, y_wrong), xytext=(x0 + 0.08, y_wrong * 0.75),
            fontsize=9, color="#C44E52")
ax.annotate(f"{y_right:,.0f}", (x0, y_right), xytext=(x0 + 0.08, y_right * 1.15),
            fontsize=9, color="#4C72B0")
ax.annotate("x2", (x0, math.sqrt(y_wrong * y_right)), xytext=(x0 - 0.42, math.sqrt(y_wrong * y_right)),
            fontsize=13, fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=1.2))
ax.text(0.52, 2 * n_one_sample(0.005) * 0.55,
        "кейс shelter_analytics:\nпосчитали 8 711 вместо 17 422 —\nмощность 50.8% вместо 80%",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="#fff3cd"))
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_3_1_mde_curve.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_3_1_mde_curve.png")

# %% [markdown]
# Шаг 3. Сплит r при ФИКСИРОВАННОМ трафике N = 20 000 юзеров, эффект +1.3 п.п.
# Аналитика: SE^2 = sigma^2 * (1/n1 + 1/n2); симуляция: 800 прогонов на точку.

# %%
N_TOTAL, MDE_SPLIT, REPS_SPLIT = 20_000, 0.013, 800
splits = (0.5, 0.4, 0.3, 0.2, 0.1)

print("=" * 78)
print("2) МОЩНОСТЬ ОТ СПЛИТА (всего %d юзеров, эффект +%.1f п.п.)" % (N_TOTAL, MDE_SPLIT * 100))
print("-" * 78)
print(f"   {'сплит':>9} | {'мощн.теория':>11} | {'мощн.симуляция':>14} | "
      f"{'трафик x':>8} | {'SE x против 50/50':>17}")
se50 = SIGMA * math.sqrt(2 / (N_TOTAL / 2))
rows = []
for r in splits:
    n1, n2 = N_TOTAL * r, N_TOTAL * (1 - r)
    se = SIGMA * math.sqrt(1 / n1 + 1 / n2)
    pw_th = stats.norm.cdf(MDE_SPLIT / se - Z_A)
    a = RNG.binomial(1, P0, size=(REPS_SPLIT, int(n1))).astype(float)
    b = RNG.binomial(1, P0 + MDE_SPLIT, size=(REPS_SPLIT, int(n2))).astype(float)
    pw_sim = float(np.mean(stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue < ALPHA))
    infl = 0.25 / (r * (1 - r))
    rows.append((r, pw_th, pw_sim, infl, se / se50))
    print(f"   {int(r*100)}/{int((1-r)*100):>3} | {pw_th:>11.1%} | {pw_sim:>14.1%} | "
          f"{infl:>8.2f} | {se/se50:>17.2f}")

rr = np.linspace(0.05, 0.5, 120)
se_curve = SIGMA * np.sqrt(1 / (N_TOTAL * rr) + 1 / (N_TOTAL * (1-rr)))
pw_curve = stats.norm.cdf(MDE_SPLIT / se_curve - Z_A) + stats.norm.cdf(-MDE_SPLIT / se_curve - Z_A)

fig, ax = plt.subplots(figsize=(8.8, 4.5))
ax.plot(rr * 100, pw_curve, "-", lw=2.4, color="#4C72B0", label="теория")
ax.plot([x[0] * 100 for x in rows], [x[2] for x in rows], "o", ms=7, color="#55A868",
        label="симуляция (800 прогонов)")
for r, _, pw_sim, _, _ in rows:
    ax.annotate(f"{pw_sim:.0%}", (r * 100, pw_sim), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=9)
ax.axhline(0.8, color="gray", ls=":", lw=1.4)
ax.text(45.5, 0.815, "power 80%", fontsize=9, color="gray")
ax.set_xlabel("доля теста r, % (контроль = 100 - r)")
ax.set_ylabel("мощность")
ax.set_title("При равных дисперсиях и том же трафике 50/50 мощнее:\n"
             "перекос 10/90 стоит x2.78 трафика (или -0.4 мощности в этой точке)")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_3_1_split_power.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_3_1_split_power.png")

# %% [markdown]
# Шаг 4. Длительность: нужно 33 154 юзеров (16 577 x 2 при MDE +1 п.п.),
# трафик 1 500/будень и 2 100/выходной. Наивно: 33 154 / 1671 = 20 дней.
# Честно: целыми неделями — 3 полные недели.

# %%
WEEKDAY, WEEKEND = 1500, 2100
NEED = 33_154
daily = np.array([WEEKDAY] * 5 + [WEEKEND] * 2)
daily = np.tile(daily, 6)[:45]  # понедельник старта, 45 дней запаса
cum = np.cumsum(daily)
day_naive = int(np.searchsorted(cum, NEED) + 1)
weeks_full = int(np.ceil(day_naive / 7))

print("=" * 78)
print("3) СКОЛЬКО ДНЕЙ ЖДАТЬ (потребность %s юзеров, MDE +1 п.п.)" % f"{NEED:,}")
print("-" * 78)
print(f"   средний день     : {daily.mean():,.0f} юзеров")
print(f"   наивный расчёт   : {NEED / daily.mean():.1f} дней -> «примерно 20»")
print(f"   по календарю     : {day_naive}-й день (это "
      f"{'будни' if daily[day_naive-1]==WEEKDAY else 'выходные'})")
print(f"   полными неделями : {weeks_full} недели = {weeks_full*7} дней, наберём "
      f"{cum[weeks_full*7-1]:,}")
print(f"   к концу 20-го дня: {cum[19]:,} юзеров — {'ХВАТАЕТ' if cum[19]>=NEED else 'НЕ хватает'}")

fig, ax = plt.subplots(figsize=(8.8, 4.4))
days = np.arange(1, len(daily) + 1)
ax.bar(days, daily, color=["#8172B3" if d > WEEKDAY else "#4C72B0" for d in daily],
       alpha=0.55, label="дневной трафик (выходные x1.4)")
ax2 = ax.twinx()
ax2.plot(days, cum, "-", lw=2.2, color="#55A868", label="накопительно")
ax2.axhline(NEED, color="#C44E52", ls="--", lw=1.8, label=f"потребность {NEED:,}")
for w in (1, 2, 3, 4, 5, 6):
    ax.axvline(7 * w + 0.5, color="gray", ls=":", lw=1)
ax2.annotate(f"полные недели: {weeks_full} нед. = {cum[weeks_full*7-1]:,}",
             (weeks_full * 7, cum[weeks_full * 7 - 1]), xytext=(weeks_full * 7 - 22, cum[weeks_full * 7 - 1] + 4200),
             fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))
ax2.annotate(f"наивно «20 дней»: {cum[19]:,} < {NEED:,}",
             (20, cum[19]), xytext=(22, cum[19] - 7000), fontsize=9,
             arrowprops=dict(arrowstyle="->", lw=1), color="#C44E52")
ax.set_xlabel("день от старта (понедельник)")
ax.set_ylabel("юзеров в день")
ax2.set_ylabel("накоплено юзеров")
ax.set_title("Длительность кроится целыми неделями: выходные не равны будням")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_3_1_calendar.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_3_1_calendar.png")

# %%
print()
print("=" * 78)
print("ГЛАВНОЕ (что должны увидеть):")
print("1) Формула n = 2(z_a+z_b)^2 sigma^2 / delta^2 на группу честная: симуляция")
print("   на расчётном n даёт ~80% мощности при любом MDE.")
print("2) Забытый x2 = половина выборки = мощность ~50.8% при целевых 80% —")
print("   для нормальной аппроксимации (Phi((z_a+z_b)/sqrt(2)-z_a)); это и есть кейс 8 711/17 422.")
print("3) Сплит: при том же трафике 50/50 -> 0.81; 30/70 -> 0.74; 20/80 -> 0.62;")
print("   10/90 -> 0.40. Возврат мощности при 10/90 стоит x2.78 трафика.")
print("4) Длительность: «20 дней» по среднему не набирает потребность (33 000 <")
print("   33 156); проектируем 3 полные недели — сезонность выходит ровно.")
