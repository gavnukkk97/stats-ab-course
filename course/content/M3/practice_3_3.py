# -*- coding: utf-8 -*-
"""Практика 3.3 — Ratio-метрики (средний чек): наивный t, бутстрап юзерами,
дельта-метод, линеаризация (кейс «ЕдаДома»).

Принцип курса: не верь формуле — проверь симуляцией.

Модель: 2000 юзеров на группу; m_i = 1 + Poisson(2) заказов; чек логнормальный
с юзер-эффектом («уровень кошелька»): чек = exp(mu + u_i + e_ij).
X_i = выручка юзера, Y_i = число заказов. Глобальный чек R = sum(X)/sum(Y).

Что делаем:
1) демо: один A/A и один A/B (+8% к чеку) — p-value всех четырёх способов;
2) A/A (500 прогонов): уровень — доля p<0.05 четырёх способов;
3) A/B (500 прогонов): мощность четырёх способов;
4) согласие: корреляция p-value дельта-метода / линеаризации / бутстрапа.

Запуск:  python3 practice_3_3.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# Шаг 0. Импорты и генератор данных: юзеры -> (X_i, Y_i).

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

MU_LN = math.log(700.0)   # базовый чек (медиана логнормального)
TAU, S_EPS = 0.6, 0.5     # sd юзер-эффекта и шума заказа (log-шкала)
LAMBDA_ORDERS = 2         # m_i = 1 + Poisson(2), среднее 3 заказа
N_USERS = 2_000           # юзеров на группу
N_SIMS = 500              # виртуальных тестов на исследование
B_BOOT = 400              # бутстрап-итераций на тест
ALPHA = 0.05


def gen_users(n_users, effect_mult=1.0, rng=None):
    """Юзер -> (X_i выручка, Y_i заказы). effect_mult умножает чек."""
    rng = rng or RNG
    m = 1 + rng.poisson(LAMBDA_ORDERS, size=n_users)
    m_max = int(m.max())
    u = rng.normal(0.0, TAU, size=(n_users, 1))
    e = rng.normal(0.0, S_EPS, size=(n_users, m_max))
    mask = np.arange(m_max)[None, :] < m[:, None]
    checks = np.exp(MU_LN + u + e) * effect_mult * mask
    return checks, m.astype(float)


def unpack(checks, m):
    X = checks.sum(axis=1)
    Y = m
    return X, Y


# --- четыре способа --------------------------------------------------------
def p_naive_orders(ca, ma, cb, mb):
    """(а) НЕКОРРЕКТНО: наивный t-тест по всем заказам как по независимым."""
    a = ca.ravel()
    a = a[a > 0]
    b = cb.ravel()
    b = b[b > 0]
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def p_bootstrap_users(ca, ma, cb, mb, n_boot=B_BOOT, rng=None):
    """(б) Бутстрап юзерами: ресемплируем пару (X_i, Y_i) целиком,
    пересчитываем глобальный чек; p-value с рецентрированием."""
    rng = rng or RNG
    xa, ya = unpack(ca, ma)
    xb, yb = unpack(cb, mb)
    na, nb = len(xa), len(xb)
    ra, rb = xa.sum() / ya.sum(), xb.sum() / yb.sum()
    d_obs = rb - ra
    idx_a = rng.integers(0, na, size=(n_boot, na))
    idx_b = rng.integers(0, nb, size=(n_boot, nb))
    ra_b = xa[idx_a].sum(axis=1) / ya[idx_a].sum(axis=1)
    rb_b = xb[idx_b].sum(axis=1) / yb[idx_b].sum(axis=1)
    d_boot = (rb_b - ra_b) - d_obs          # рецентрирование под H0
    return float(np.mean(np.abs(d_boot) >= abs(d_obs)))


def ratio_delta_stats(checks, m):
    """Глобальный чек и Var(R) дельта-методом для одной группы."""
    x, y = unpack(checks, m)
    n = len(x)
    xb, yb = x.mean(), y.mean()
    r = xb / yb
    var_x, var_y = x.var(ddof=1), y.var(ddof=1)
    cov_xy = np.mean((x - xb) * (y - yb)) * n / (n - 1)
    var_r = (var_x + r**2 * var_y - 2 * r * cov_xy) / (n * yb**2)
    return r, var_r, xb, yb


def p_delta_method(ca, ma, cb, mb):
    """(в) Дельта-метод: Var(R) через дисперсии/ковариацию на юзере."""
    ra, va, _, _ = ratio_delta_stats(ca, ma)
    rb, vb, _, _ = ratio_delta_stats(cb, mb)
    z = (rb - ra) / math.sqrt(va + vb)
    return 2 * stats.norm.sf(abs(z))


def p_linearization(ca, ma, cb, mb):
    """(г) Линеаризация: L_i = X_i - R_контр*Y_i, t-тест Уэлча по L."""
    xa, ya = unpack(ca, ma)
    xb, yb = unpack(cb, mb)
    r_ctrl = xa.sum() / ya.sum()
    la, lb = xa - r_ctrl * ya, xb - r_ctrl * yb
    return stats.ttest_ind(la, lb, equal_var=False).pvalue


METHODS = (("naive", "наивный t по заказам (НЕКОРРЕКТНО)", p_naive_orders),
           ("boot", "бутстрап юзерами", p_bootstrap_users),
           ("delta", "дельта-метод", p_delta_method),
           ("lin", "линеаризация + t", p_linearization))


def run_sims(n_sims, effect_mult):
    """Прогоняет n_sims виртуальных тестов -> dict массивов p-value."""
    out = {k: [] for k, _, _ in METHODS}
    for _ in range(n_sims):
        ca, ma = gen_users(N_USERS, 1.0)
        cb, mb = gen_users(N_USERS, effect_mult)
        for k, _, fn in METHODS:
            out[k].append(fn(ca, ma, cb, mb))
    return {k: np.asarray(v) for k, v in out.items()}


# %% [markdown]
# Шаг 1. Демо: один A/A и один A/B (+8% к чеку) — свои seed, чтобы примеры
# были воспроизводимы. Смотрим на глобальный чек, среднее юзерских чеков
# и p-value четырёх способов.

# %%
def demo(effect_mult, title, seed):
    ca, ma = gen_users(N_USERS, 1.0, rng=np.random.default_rng(seed))
    cb, mb = gen_users(N_USERS, effect_mult, rng=np.random.default_rng(seed + 1))
    ra, va, xba, yba = ratio_delta_stats(ca, ma)
    rb, vb, xbb, ybb = ratio_delta_stats(cb, mb)
    mean_of_means_a = np.mean((ca.sum(axis=1) / ma))
    mean_of_means_b = np.mean((cb.sum(axis=1) / mb))
    print("=" * 78)
    print("ДЕМО — %s (n = %s юзеров на группу)" % (title, f"{N_USERS:,}"))
    print("-" * 78)
    print(f"   глобальный чек  A -> B      : {ra:7.1f} -> {rb:7.1f} руб "
          f"({(rb - ra) / ra:+.2%})")
    print(f"   среднее юзерских чеков A->B : {mean_of_means_a:7.1f} -> {mean_of_means_b:7.1f} руб "
          f"(ДРУГАЯ метрика!)")
    print(f"   заказов на юзера A / B      : {yba:.2f} / {ybb:.2f}")
    print(f"   {'способ':<34} {'p-value':>10}")
    for k, label, fn in METHODS:
        print(f"   {label:<34} {fn(ca, ma, cb, mb, rng=np.random.default_rng(seed + 2)) if k == 'boot' else fn(ca, ma, cb, mb):>10.4f}")
    return ca, ma, cb, mb


demo(1.0, "A/A-ТЕСТ (эффекта нет)", seed=11)
print()
demo(1.08, "A/B-ТЕСТ (+8% к чеку)", seed=21)

# %% [markdown]
# Шаг 2. A/A: уровень значимости четырёх способов (500 прогонов).
# Ожидание: наивный сломан (зависимость заказов юзера), остальные ~5%.

# %%
p_aa = run_sims(N_SIMS, 1.0)

print("=" * 78)
print("2) A/A: УРОВЕНЬ ЗНАЧИМОСТИ (%d прогонов)" % N_SIMS)
print("-" * 78)
for k, label, _ in METHODS:
    print(f"   {label:<34}: доля p<0.05 = {np.mean(p_aa[k] < ALPHA):5.1%}")
print(f"   {'номинал alpha':<34}: {'':>16}= 5.0%")

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.7), sharey=True)
for ax, (k, label, _) in zip(axes, METHODS):
    ax.hist(p_aa[k], bins=36, color="#C44E52" if k == "naive" else "#4C72B0",
            alpha=0.85)
    ax.axhline(N_SIMS / 36, color="gray", ls="--", lw=1.2)
    ax.set_title(f"{label}\np<0.05 в {np.mean(p_aa[k] < ALPHA):.0%} прогонов",
                 fontsize=9.5)
    ax.set_xlabel("p-value")
axes[0].set_ylabel("число прогонов")
fig.suptitle("A/A на ratio-метрике (средний чек): наивный t по заказам ломает уровень",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "practice_3_3_aa_level.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_3_3_aa_level.png")

# %% [markdown]
# Шаг 3. A/B с эффектом +8% к чеку: мощность четырёх способов.
# «Мощность» наивного куплена сломанным уровнем — сравниваем с оговоркой.

# %%
p_ab = run_sims(N_SIMS, 1.08)

print("=" * 78)
print("3) A/B (+8%% К ЧЕКУ): МОЩНОСТЬ (%d прогонов)" % N_SIMS)
print("-" * 78)
print(f"   {'способ':<34} | {'мощность':>8} | {'уровень в A/A':>13}")
for k, label, _ in METHODS:
    print(f"   {label:<34} | {np.mean(p_ab[k] < ALPHA):>8.1%} | "
          f"{np.mean(p_aa[k] < ALPHA):>13.1%}")

fig, ax = plt.subplots(figsize=(9.2, 4.4))
x = np.arange(len(METHODS))
lvl = [np.mean(p_aa[k] < ALPHA) for k, _, _ in METHODS]
pwr = [np.mean(p_ab[k] < ALPHA) for k, _, _ in METHODS]
ax.bar(x - 0.2, lvl, 0.38, color="#C44E52", label="A/A: фактический уровень")
ax.bar(x + 0.2, pwr, 0.38, color="#4C72B0", label="A/B (+8% к чеку): мощность")
ax.axhline(ALPHA, color="k", ls=":", lw=1.5)
ax.text(len(METHODS) - 0.55, 0.058, "номинал 5%", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["наивный t\nпо заказам", "бутстрап\nюзерами",
                    "дельта-\nметод", "линеаризация\n+ t"], fontsize=10)
for i, (l, p) in enumerate(zip(lvl, pwr)):
    ax.text(i - 0.2, l + 0.012, f"{l:.0%}", ha="center", fontsize=9, color="#C44E52")
    ax.text(i + 0.2, p + 0.012, f"{p:.0%}", ha="center", fontsize=9, color="#4C72B0")
ax.set_ylabel("доля p < 0.05")
ax.set_title("Средний чек: три корректных способа согласованы по уровню и мощности")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "practice_3_3_ab_power.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_3_3_ab_power.png")

# %% [markdown]
# Шаг 4. Согласие способов: корреляция p-value на A/A. Дельта-метод и
# линеаризация — одна статистика в двух упаковках; бутстрап рядом.

# %%
ok = (p_aa["delta"] > 0) & (p_aa["boot"] > 0) & (p_aa["lin"] > 0)
r_dl = stats.pearsonr(p_aa["delta"][ok], p_aa["lin"][ok])[0]
r_db = stats.pearsonr(p_aa["delta"][ok], p_aa["boot"][ok])[0]
med_ratio = np.median(p_aa["lin"][ok] / p_aa["delta"][ok])

print("=" * 78)
print("4) СОГЛАСЕ СПОСОБОВ НА A/A (корреляция p-value, %d прогонов)" % N_SIMS)
print("-" * 78)
print(f"   дельта-метод vs линеаризация : r = {r_dl:.4f}; "
      f"медиана отношения p_lin/p_delta = {med_ratio:.3f}")
print(f"   дельта-метод vs бутстрап     : r = {r_db:.4f}")

fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.scatter(p_aa["delta"][ok], p_aa["lin"][ok], s=12, alpha=0.55,
           color="#4C72B0", label=f"линеаризация vs дельта: r={r_dl:.3f}")
ax.scatter(p_aa["delta"][ok], p_aa["boot"][ok], s=12, alpha=0.45,
           color="#55A868", label=f"бутстрап vs дельта: r={r_db:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1.2)
ax.set_xlabel("p-value дельта-метода")
ax.set_ylabel("p-value конкурента")
ax.set_title("Три корректных способа дают почти одинаковые p-value (A/A)")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(HERE / "practice_3_3_agreement.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_3_3_agreement.png")

# %%
print()
print("=" * 78)
print("ГЛАВНОЕ (что должны увидеть):")
print("1) Наивный t по заказам в A/A даёт долю p<0.05 заметно выше 5% —")
print("   заказы юзера зависимы, наблюдений больше, чем информации.")
print("2) Бутстрап юзерами, дельта-метод и линеаризация держат уровень ~5%")
print("   и дают согласованную мощность в A/B.")
print("3) Глобальный чек и среднее юзерских чеков — разные числа")
print("   (в A/B меняются по-разному): решать можно только по нужной бизнесу")
print("   метрике.")
print("4) Дельта-метод и линеаризация — практически одна статистика;")
print("   линеаризация дополнительно даёт поюзерный сигнал для CUPED (урок 3.5).")
