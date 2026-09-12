# -*- coding: utf-8 -*-
"""Практика 3.2 — Юнит рандомизации и анализа: зависимость наблюдений,
ICC, design effect, кластерные SE (кейс «ЕдаДома»).

Принцип курса: не верь формуле — проверь симуляцией.

Модель: юзер делает m_i = 1 + Poisson(5) заказов (в среднем 6);
чек = exp(u_i + e_ij), u_i ~ N(0, tau^2) — «уровень кошелька» юзера,
e_ij ~ N(0, s^2) — шум заказа. Заказы одного юзера коррелируют (ICC).

Что делаем:
1) A/A (2000 прогонов, 500 юзеров на группу): доля p<0.05 у
   (а) наивного t по ЗАКАЗАМ, (б) t по юзерским средним,
   (в) кластерных SE (ratio-оценщик по кластерным суммам);
2) оценка ICC из данных (ANOVA) против истинного значения;
3) design effect и эффективный n для своих данных;
4) A/B с эффектом +3% к чеку: мощность трёх критериев.

Запуск:  python3 practice_3_2.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# Шаг 0. Импорты и генератор данных: юзеры -> заказы с внутрикластерной
# корреляцией. ICC для ln N: tau^2/(tau^2+s^2); для уровней чека
# corr = (exp(tau^2)-1)/(exp(tau^2+s^2)-1) — посчитаем и сверим.

# %%
import math

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RNG = np.random.default_rng(20260908)
HERE = Path(__file__).resolve().parent

TAU, S_EPS = 0.5, 0.5          # sd юзер-эффекта и шума заказа (log-шкала)
MU_LN = math.log(850.0)        # базовый чек ~850 руб.
LAMBDA_ORDERS = 5              # m_i = 1 + Poisson(5), среднее 6
N_USERS = 500                  # юзеров на группу
N_AA, N_AB = 2_000, 1_000      # число виртуальных тестов
ALPHA = 0.05
ICC_TRUE_LEVELS = (math.exp(TAU**2) - 1) / (math.exp(TAU**2 + S_EPS**2) - 1)


def gen_group(n_users, effect_mult=1.0, rng=None):
    """Юзеры -> (массив всех чеков, массив m_i). effect_mult умножает чек."""
    rng = rng or RNG
    m = 1 + rng.poisson(LAMBDA_ORDERS, size=n_users)
    m_max = int(m.max())
    u = rng.normal(0.0, TAU, size=(n_users, 1))
    e = rng.normal(0.0, S_EPS, size=(n_users, m_max))
    mask = np.arange(m_max)[None, :] < m[:, None]
    checks = np.exp(MU_LN + u + e) * effect_mult * mask
    return checks, m


def p_naive_by_orders(checks_a, m_a, checks_b, m_b):
    """НЕКОРРЕКТНЫЙ критерий: t-тест по всем заказам как по независимым."""
    a = checks_a.ravel()
    a = a[a > 0]
    b = checks_b.ravel()
    b = b[b > 0]
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def p_by_user_means(checks_a, m_a, checks_b, m_b):
    """Корректный: t-тест на юзерских средних (юнит анализа = юнит рандомизации)."""
    ma = checks_a.sum(axis=1) / m_a
    mb = checks_b.sum(axis=1) / m_b
    return stats.ttest_ind(ma, mb, equal_var=False).pvalue


def theta_cluster(checks, m):
    """Кластерный ratio-оценщик: theta = сумма чеков / число заказов."""
    S, M = checks.sum(), m.sum()
    return S, M


def p_cluster_se(checks_a, m_a, checks_b, m_b):
    """Кластерные SE: z = (theta_A - theta_B)/sqrt(SE_A^2 + SE_B^2),
    SE^2 = sum_i (S_i - theta*m_i)^2 / M^2."""
    z_list, thetas = [], []
    for checks, m in ((checks_a, m_a), (checks_b, m_b)):
        S_i = checks.sum(axis=1)
        M = m.sum()
        theta = S_i.sum() / M
        se2 = np.sum((S_i - theta * m) ** 2) / M**2
        z_list.append((theta, se2))
        thetas.append(theta)
    se = math.sqrt(z_list[0][1] + z_list[1][1])
    z = (thetas[0] - thetas[1]) / se
    return 2 * stats.norm.sf(abs(z))


def run_sims(n_sims, effect_mult):
    """Прогоняет n_sims виртуальных A/A (effect_mult=1) или A/B тестов."""
    p = {"orders": [], "users": [], "cluster": []}
    for _ in range(n_sims):
        ca, ma = gen_group(N_USERS, 1.0)
        cb, mb = gen_group(N_USERS, effect_mult)
        p["orders"].append(p_naive_by_orders(ca, ma, cb, mb))
        p["users"].append(p_by_user_means(ca, ma, cb, mb))
        p["cluster"].append(p_cluster_se(ca, ma, cb, mb))
    return {k: np.asarray(v) for k, v in p.items()}


# %% [markdown]
# Шаг 1. A/A-тесты: уровень трёх критериев. Наивный t по заказам считает
# наблюдений в ~6 раз больше, чем есть независимых юнитов.

# %%
p_aa = run_sims(N_AA, 1.0)

labels = {"orders": "наивный t по заказам (НЕКОРРЕКТНО)",
          "users": "t по юзерским средним",
          "cluster": "кластерные SE"}

print("=" * 78)
print("1) A/A-ТЕСТ: уровень значимости трёх критериев (%d прогонов)" % N_AA)
print("-" * 78)
for k in ("orders", "users", "cluster"):
    print(f"   {labels[k]:<42}: доля p<0.05 = {np.mean(p_aa[k] < ALPHA):5.1%}")
print(f"   {'номинал alpha':<42}: {'':>14} = 5.0%")

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
for ax, k in zip(axes, ("orders", "users", "cluster")):
    ax.hist(p_aa[k], bins=40, color="#4C72B0" if k == "orders" else "#55A868",
            alpha=0.85)
    ax.axhline(N_AA / 40, color="#C44E52", ls="--", lw=1.3)
    ax.set_title(f"{labels[k]}\nдоля p<0.05 = {np.mean(p_aa[k] < ALPHA):.1%}",
                 fontsize=10)
    ax.set_xlabel("p-value")
axes[0].set_ylabel("число прогонов")
fig.suptitle("A/A на зависимых данных: наивный t по заказам ломает уровень; "
             "юзер/кластер — держат 5%", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "practice_3_2_aa_levels.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_3_2_aa_levels.png")

# %% [markdown]
# Шаг 2. Оценка ICC из данных: однофакторная модель со случайным эффектом
# юзера. ICC = (MSB - MSW) / (MSB + (m0 - 1)*MSW), m0 — эффективный средний
# размер кластера. Сверяем с истинным значением генератора.

# %%
def estimate_icc(checks, m):
    y = checks.ravel()
    y = y[y > 0]
    n_total = len(y)
    k = len(m)
    user_means = checks.sum(axis=1) / m
    grand = y.mean()
    msb = np.sum(m * (user_means - grand) ** 2) / (k - 1)
    ss_within = np.sum((checks - user_means[:, None]) ** 2 * (checks > 0))
    msw = ss_within / (n_total - k)
    m0 = (n_total - np.sum(m**2) / n_total) / (k - 1)
    return (msb - msw) / (msb + (m0 - 1) * msw)


icc_hats = []
for _ in range(50):
    c, m = gen_group(N_USERS * 2)
    icc_hats.append(estimate_icc(c, m))
icc_hat_mean = float(np.mean(icc_hats))

print("=" * 78)
print("2) ОЦЕНКА ICC ПО ДАННЫМ (50 повторов, ANOVA-оценка)")
print("-" * 78)
print(f"   истинная ICC (для уровней чека):   {ICC_TRUE_LEVELS:.3f}")
print(f"   оценка по данным (среднее из 50):  {icc_hat_mean:.3f} "
      f"(SD повторов {np.std(icc_hats):.3f})")
print(f"   заказов на юзера в среднем:        {1 + LAMBDA_ORDERS}")

# %% [markdown]
# Шаг 3. Design effect и эффективный n на ОДНОЙ реализации данных
# (контрольная группа): DEFF = 1 + (m-1)*ICC; эффективных наблюдений
# n_eff = n_orders / DEFF. Сравниваем с «наивным» числом строк.

# %%
c0, m0_arr = gen_group(N_USERS * 2)
n_orders = int(m0_arr.sum())
m_bar = m0_arr.mean()
icc0 = estimate_icc(c0, m0_arr)
deff = 1 + (m_bar - 1) * icc0
n_eff = n_orders / deff

# эмпирическая проверка DEFF: отношение честного SE (по юзерам) к наивному
user_means0 = c0.sum(axis=1) / m0_arr
orders0 = c0.ravel()
orders0 = orders0[orders0 > 0]
se_naive = orders0.std(ddof=1) / math.sqrt(n_orders)
se_honest = user_means0.std(ddof=1) / math.sqrt(len(m0_arr)) * math.sqrt(m_bar)
# SE глобального среднего на заказ ~ SE юзерских средних * sqrt(m_bar):
# среднее на заказ = m_bar * среднее юзерских средних при равных m

print("=" * 78)
print("3) DESIGN EFFECT И ЭФФЕКТИВНЫЙ n (одна реализация, %d юзеров)" % len(m0_arr))
print("-" * 78)
print(f"   заказов всего / на юзера        : {n_orders:,} / {m_bar:.1f}")
print(f"   ICC по данным                   : {icc0:.3f}")
print(f"   DEFF = 1 + (m-1)*ICC            : {deff:.2f}")
print(f"   эффективных наблюдений из заказов: {n_eff:,.0f} "
      f"(в {n_orders / n_eff:.1f} раза меньше строк датасета)")
print(f"   предсказанное завышение SE      : x{math.sqrt(deff):.2f}")

# %% [markdown]
# Шаг 4. A/B с реальным эффектом +3% к чеку: мощность трёх критериев.
# Наивный выглядит «мощнее», но его уровень сломан (27% вместо 5%) —
# сравниваем при одинаковом честном пороге.

# %%
p_ab = run_sims(N_AB, 1.03)

print("=" * 78)
print("4) A/B С ЭФФЕКТОМ +3%% К ЧЕКУ: мощность (%d прогонов)" % N_AB)
print("-" * 78)
print(f"   {'критерий':<42} | {'мощность':>8} | {'уровень в A/A':>13}")
for k in ("orders", "users", "cluster"):
    print(f"   {labels[k]:<42} | {np.mean(p_ab[k] < ALPHA):>8.1%} | "
          f"{np.mean(p_aa[k] < ALPHA):>13.1%}")

fig, ax = plt.subplots(figsize=(8.6, 4.3))
methods = ("orders", "users", "cluster")
x = np.arange(3)
lvl = [np.mean(p_aa[k] < ALPHA) for k in methods]
pwr = [np.mean(p_ab[k] < ALPHA) for k in methods]
ax.bar(x - 0.2, lvl, 0.38, color="#C44E52", label="A/A: фактический уровень")
ax.bar(x + 0.2, pwr, 0.38, color="#4C72B0", label="A/B (+3% к чеку): мощность")
ax.axhline(0.05, color="k", ls=":", lw=1.5)
ax.text(2.42, 0.065, "номинал 5%", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["наивный t\nпо заказам", "t по юзерским\nсредним", "кластерные SE"],
                   fontsize=10)
for i, (l, p) in enumerate(zip(lvl, pwr)):
    ax.text(i - 0.2, l + 0.008, f"{l:.0%}", ha="center", fontsize=9, color="#C44E52")
    ax.text(i + 0.2, p + 0.008, f"{p:.0%}", ha="center", fontsize=9, color="#4C72B0")
ax.set_ylabel("доля p < 0.05")
ax.set_title("Зависимые заказы: «мощность» наивного критерия куплена сломанным уровнем")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "practice_3_2_power.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_3_2_power.png")

# %%
print()
print("=" * 78)
print("ГЛАВНОЕ (что должны увидеть):")
print("1) Наивный t по заказам в A/A: доля p<0.05 ~25-30% вместо 5% —")
print("   наблюдений в 6 раз больше, чем независимых юнитов (юзеров).")
print("2) t по юзерским средним и кластерные SE держат уровень ~5%:")
print("   анализ на юните рандомизации = правильная неопределённость.")
print("3) ICC оценивается из данных ANOVA-оценкой и совпадает с истинной;")
print("   DEFF = 1 + (m-1)*ICC ~ 3.2: 3000 заказов стоят ~950 наблюдений.")
print("4) В A/B «мощность» наивного куплена ложными срабатываниями;")
print("   корректные критерии при одинаковом альфа сопоставимы по мощности.")
