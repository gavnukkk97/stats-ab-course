# -*- coding: utf-8 -*-
"""Практика 3.5 — CUPED и семейство снижения дисперсии (кейс «ЕдаДома»).

Не верь — проверь симуляцией.

Что делаем:
1) синтетика: пре-метрика и метрика с корреляцией rho = 0.6 / 0.8;
   SE raw против SE CUPED — сжатие должно быть 1/sqrt(1-rho^2);
   то же в мощности и в «днях теста» (экономия наблюдений = rho^2);
2) CUPED руками: theta = Cov(X,Y)/Var(X) на объединённых группах,
   Y_cv = Y - theta*X, t-test; потом та же оценка через OLS
   (Y ~ 1 + T + X) — сверка до пятого знака: CUPED = регрессия = ANCOVA;
3) пост-стратификация по известной страте (город): взвешенное среднее,
   регрессия с дамми страт и CUPED с непрерывной пре-метрикой — кто кого;
4) A/A-уровень CUPED: доля p<0.05 ~ 5%, p-value равномерен;
5) НЕправильный CUPED с пост-тритмент ковариатой («метрика первой недели
   теста»): симуляция показывает, как ковариата съедает ~половину эффекта.

Запуск:  python3 practice_3_5.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# # Практика 3.5 — CUPED: вычитаем предсказуемое
# Формула урока: Y_cv = Y − θ·X, θ = Cov(X,Y)/Var(X),
# Var(Y_cv) = Var(Y)·(1 − ρ²) — это знакомая формула дисперсии разности
# из 2.2 (s_d² = s_x² + s_y² − 2·r·s_x·s_y), только с оптимальным θ.

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(35)  # seed -> результат воспроизводим
HERE = Path(__file__).resolve().parent
ALPHA = 0.05


def world(n, rho, delta, rng):
    """Один мир: сплит 50/50, эффект delta в тесте.

    X — пре-метрика (измерена ДО теста), Y — метрика эксперимента.
    Var(X) = Var(Y) = 1, corr(X, Y) = rho — конструкция чистая для сверки с теорией.
    """
    X = rng.normal(0, 1, n)
    Y0 = rho * X + rng.normal(0, np.sqrt(1 - rho**2), n)
    T = rng.integers(0, 2, n)
    return X, Y0 + delta * T, T


def cuped(X, Y, T):
    """CUPED: theta на объединённых группах, разность средних Y_cv."""
    theta = np.cov(X, Y)[0, 1] / X.var()
    Ycv = Y - theta * X
    return Ycv[T == 1].mean() - Ycv[T == 0].mean(), theta


# %% [markdown]
# ## Шаг 1. Сжатие шума: SE raw против SE CUPED при ρ = 0.6 и 0.8
# Эффект δ = 0.1 (в SD метрики). Теория обещает: SE сжимается в 1/√(1−ρ²)
# раза (1.25 и 1.67), а нужный n падает в 1/(1−ρ²) раза — экономия ρ².

# %%
DELTA, REPS = 0.10, 400
rows = []
for rho in (0.6, 0.8):
    est_raw, est_cup = [], []
    for _ in range(REPS):
        X, Y, T = world(5_000, rho, DELTA, RNG)
        est_raw.append(Y[T == 1].mean() - Y[T == 0].mean())
        d, _ = cuped(X, Y, T)
        est_cup.append(d)
    se_r, se_c = np.std(est_raw), np.std(est_cup)
    rows.append({
        "ρ": rho,
        "SE raw": se_r,
        "SE CUPED": se_c,
        "сжатие": se_r / se_c,
        "теория 1/√(1−ρ²)": 1 / np.sqrt(1 - rho**2),
        "экономия n = ρ²": f"{rho**2:.0%}",
        "смещение": np.mean(est_cup) - DELTA,
    })
tab1 = pd.DataFrame(rows).set_index("ρ")

print("=" * 80)
print(f"1) SE raw против SE CUPED (δ = {DELTA}, n = 5000, {REPS} миров)")
print(tab1.round(4).to_string())
print("   Сжатие SE совпадает с 1/√(1−ρ²); смещение ~ 0 (наши ~0.001 — шум симуляции).")
for rho in (0.6, 0.8):
    n80_raw = (1.96 + 0.8416) ** 2 * 2 / DELTA ** 2
    print(f"   ρ = {rho}: для 80% мощности raw нужно n = {n80_raw:,.0f}, CUPED — {n80_raw * (1 - rho**2):,.0f}"
          f" (тест 14 дней -> {14 * (1 - rho**2):.1f} дня)".replace(",", " "))

# мощность при одном и том же n = 1000
pw_raw, pw_cup, rho = 0, 0, 0.8
for _ in range(1_000):
    X, Y, T = world(1_000, rho, DELTA, RNG)
    pw_raw += stats.ttest_ind(Y[T == 1], Y[T == 0], equal_var=False).pvalue < ALPHA
    th = np.cov(X, Y)[0, 1] / X.var()
    pw_cup += stats.ttest_ind((Y - th * X)[T == 1], (Y - th * X)[T == 0], equal_var=False).pvalue < ALPHA
print(f"   Мощность при n = 1000 (ρ = {rho}): raw = {pw_raw / 1_000:.0%} -> CUPED = {pw_cup / 1_000:.0%}")

# %%
fig, ax = plt.subplots(figsize=(8.6, 4.2))
rhos = np.linspace(0, 0.95, 100)
ax.plot(rhos, 1 / np.sqrt(1 - rhos**2), lw=2.4, color="#4C72B0", label="теория 1/√(1−ρ²)")
ax.plot(tab1.index, tab1["сжатие"], "o", ms=11, color="#C44E52", label="симуляция (SE raw / SE CUPED)")
for r, row in tab1.iterrows():
    ax.annotate(f"ρ={r}: ×{row['сжатие']:.2f}", (r, row["сжатие"]), xytext=(8, -14),
                textcoords="offset points", fontsize=10, fontweight="bold")
ax.set_xlabel("корреляция пре-метрики с метрикой, ρ")
ax.set_ylabel("во сколько раз CUPED сжимает SE")
ax.set_title("CUPED: чем сильнее «якорь» пре-периода, тем тише тест")
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "practice_3_5_se.png", dpi=150)
print("Сохранено: practice_3_5_se.png")

# %% [markdown]
# ## Шаг 2. CUPED руками и через OLS — это одна и та же оценка
# Руками: θ = Cov(X,Y)/Var(X) на объединённых группах, Y_cv = Y − θ·X,
# разность средних Y_cv. Регрессией: Y ~ 1 + T + X, коэффициент при T.
# Должны совпасть вплоть до ошибок округления — CUPED и есть ANCOVA из 2.5.

# %%
X, Y, T = world(4_000, 0.8, DELTA, RNG)
d_hand, theta = cuped(X, Y, T)

# OLS: дизайн-матрица [1, T, X]
Dm = np.column_stack([np.ones(len(Y)), T, X])
beta, *_ = np.linalg.lstsq(Dm, Y, rcond=None)
resid = Y - Dm @ beta
se_T = np.sqrt((resid @ resid) / (len(Y) - 3) * np.linalg.inv(Dm.T @ Dm)[1, 1])

# вычитание среднего из X не меняет разность средних (меняет только «ноль» шкалы)
Ycv_centered = Y - theta * (X - X.mean())
d_centered = Ycv_centered[T == 1].mean() - Ycv_centered[T == 0].mean()

print("=" * 80)
print("2) CUPED руками против OLS (ρ = 0.8, n = 4000, истинный эффект 0.10)")
print(f"   руками:  θ = {theta:.4f}, разность средних Y_cv = {d_hand:+.5f}")
print(f"   OLS:     коэф. при T = {beta[1]:+.5f}, коэф. при X = {beta[2]:.4f} (= θ),")
print(f"            SE коэф. = {se_T:.5f}, t = {beta[1] / se_T:.2f}")
print(f"   с центрированием X: разность = {d_centered:+.5f} — та же (константа вычитается в обеих группах)")
print("   CUPED = ANCOVA = регрессия с ковариатой: три имени, одна оценка (шлагбаум из 2.5).")

# %% [markdown]
# ## Шаг 3. Пост-стратификация: взвешенное среднее против CUPED
# Города (страты по пре-периоду, доли известны: 50/30/20) различаются уровнем
# трат; сплит 50/50 балансирует страты лишь в среднем. Сравниваем оценки:
# raw / пост-стратификация (взвешенное среднее по стратам) / CUPED с непрерывной
# пре-метрикой / регрессия с дамми страт (= пост-страт через OLS).

# %%
W_STRATA = np.array([0.5, 0.3, 0.2])   # генеральные доли городов (известны!)
MU_S = np.array([0.0, 0.9, 1.8])       # города различаются уровнем


def world_strat(n, delta, rng):
    s = rng.choice([0, 1, 2], n, p=W_STRATA)
    u = rng.normal(0, 1, n)                                  # латентная тяга юзера
    X = MU_S[s] + 0.8 * u + rng.normal(0, 0.6, n)            # пре-метрика
    T = rng.integers(0, 2, n)
    Y = MU_S[s] + 0.8 * u + rng.normal(0, 0.6, n) + delta * T
    return s, X, Y, T


REPS3, N3 = 400, 3_000
res3 = {"raw": [], "пост-стратификация": [], "CUPED (пре-метрика)": [], "регрессия с дамми страт": []}
for _ in range(REPS3):
    s, X, Y, T = world_strat(N3, DELTA, RNG)
    res3["raw"].append(Y[T == 1].mean() - Y[T == 0].mean())
    ps = {g: sum(W_STRATA[k] * Y[(T == g) & (s == k)].mean() for k in range(3)) for g in (0, 1)}
    res3["пост-стратификация"].append(ps[1] - ps[0])
    d, _ = cuped(X, Y, T)
    res3["CUPED (пре-метрика)"].append(d)
    Dm = np.column_stack([np.ones(N3), T] + [(s == k).astype(float) for k in (1, 2)])
    b, *_ = np.linalg.lstsq(Dm, Y, rcond=None)
    res3["регрессия с дамми страт"].append(b[1])

tab3 = pd.DataFrame({
    "SE оценки эффекта": {k: np.std(v) for k, v in res3.items()},
    "смещение": {k: np.mean(v) - DELTA for k, v in res3.items()},
    "SD-шум меньше raw": {k: 1 - np.std(v) / np.std(res3["raw"]) for k, v in res3.items()},
}).round(4)
rho_strat = np.corrcoef(*world_strat(200_000, 0, RNG)[1:3])[0, 1]

print("=" * 80)
print(f"3) Стратифицированный мир (3 города, {REPS3} миров, n = {N3}, ρ(X,Y) = {rho_strat:.2f})")
print(tab3.to_string())
print("   Пост-стратификация = CUPED с ковариатой-«стратой»: дамми-регрессия совпадает со")
print("   взвешенным средним. Непрерывная пре-метрика сильнее: в ней больше информации,")
print("   чем в трёх группах. CUPED — обобщение стратификации (shelter, гл. 5–7).")

# %% [markdown]
# ## Шаг 4. A/A-уровень CUPED
# Ковариата вычитается из обеих групп одной и той же функцией — при честном
# сплите CUPED не создаёт разницы там, где её нет: доля p<0.05 держит 5%.

# %%
REPS4 = 3_000
p_aa = []
for _ in range(REPS4):
    X, Y, T = world(2_000, 0.8, 0.0, RNG)
    th = np.cov(X, Y)[0, 1] / X.var()
    p_aa.append(stats.ttest_ind((Y - th * X)[T == 1], (Y - th * X)[T == 0], equal_var=False).pvalue)
p_aa = np.array(p_aa)
ks_p = stats.kstest(p_aa, "uniform").pvalue
print("=" * 80)
print(f"4) A/A-уровень CUPED ({REPS4:,} миров, ρ = 0.8): доля p<0.05 = {np.mean(p_aa < ALPHA):.1%}".replace(",", " "))
print(f"   p-value равномерен: KS-тест против uniform, p = {ks_p:.2f} (не отвергаем равномерность)")

# %% [markdown]
# ## Шаг 5. Неправильный CUPED: ковариата, заражённая тритментом
# «Оптимизация»: берём в качестве ковариаты метрику ПЕРВОЙ НЕДЕЛИ теста —
# она же коррелирует с метрикой (та же сессия!), и корреляция выше, чем у
# пре-периода. Но она уже содержит эффект тритмента (γ = 0.8 доли эффекта).
# Симуляция показывает, чем это кончается: оценка смещается к нулю.

# %%
DELTA5, GAMMA, N5, REPS5 = 0.20, 0.8, 4_000, 2_000
est5 = {"raw": [], "CUPED (пре-ковариата, правильно)": [], "CUPED (метрика 1-й недели, НЕПРАВИЛЬНО)": []}
for _ in range(REPS5):
    Xpre = RNG.normal(0, 1, N5)
    Y0 = 0.8 * Xpre + RNG.normal(0, 0.6, N5)
    T = RNG.integers(0, 2, N5)
    Y = Y0 + DELTA5 * T
    Cpost = GAMMA * DELTA5 * T + 0.8 * Xpre + RNG.normal(0, 0.6, N5)  # заражена эффектом
    est5["raw"].append(Y[T == 1].mean() - Y[T == 0].mean())
    for key, C in (("CUPED (пре-ковариата, правильно)", Xpre),
                   ("CUPED (метрика 1-й недели, НЕПРАВИЛЬНО)", Cpost)):
        th = np.cov(C, Y)[0, 1] / C.var()
        est5[key].append((Y - th * C)[T == 1].mean() - (Y - th * C)[T == 0].mean())

tab5 = pd.DataFrame({
    "E[оценка]": {k: np.mean(v) for k, v in est5.items()},
    "SE": {k: np.std(v) for k, v in est5.items()},
    "смещение": {k: np.mean(v) - DELTA5 for k, v in est5.items()},
}).round(4)
print("=" * 80)
print(f"5) Ковариата, заражённая тритментом (δ = {DELTA5}, γ = {GAMMA}, {REPS5:,} миров)".replace(",", " "))
print(tab5.to_string())
eaten = 1 - tab5.loc["CUPED (метрика 1-й недели, НЕПРАВИЛЬНО)", "E[оценка]"] / DELTA5
print(f"   Неправильный CUPED съел {eaten:.0%} эффекта: и оценка занижена, и SE не лучше.")
print("   Правило: ковариата измеряется ДО начала теста; всё, что после старта, — табу.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
axes[0].hist(p_aa, bins=30, color="#4C72B0", alpha=0.88)
axes[0].axhline(REPS4 / 30, color="#C44E52", ls="--", lw=1.6, label="уровень равномерности")
axes[0].set_title(f"A/A-уровень CUPED: p-value равномерен,\nдоля p<0.05 = {np.mean(p_aa < ALPHA):.1%}")
axes[0].set_xlabel("p-value")
axes[0].legend(fontsize=9)
for k, v in est5.items():
    axes[1].hist(v, bins=70, alpha=0.65, label=f"{k.split(' (')[0]}: E={np.mean(v):.3f}")
axes[1].axvline(DELTA5, color="black", ls="--", lw=1.8, label=f"истинный эффект = {DELTA5}")
axes[1].set_xlabel("оценка эффекта")
axes[1].set_title("Ковариата из эксперимента смещает оценку к нулю;\nпре-периодная — несмещённа и точнее")
axes[1].legend(fontsize=8)
fig.suptitle("Границы CUPED: честен в A/A, но только с пре-периодной ковариатой", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "practice_3_5_bias.png", dpi=150)
print("Сохранено: practice_3_5_bias.png")

# %%
print()
print("=" * 80)
print("ГЛАВНОЕ (что должны увидеть):")
print(f"1) SE сжатие: ρ=0.6 -> ×{tab1.loc[0.6, 'сжатие']:.2f}, ρ=0.8 -> ×{tab1.loc[0.8, 'сжатие']:.2f}"
      f" (теория 1/√(1−ρ²)); экономия наблюдений = ρ² (36%/64%) — тест 14 дней -> 9/5 дней.")
print(f"2) Мощность при n=1000: raw {pw_raw/1_000:.0%} -> CUPED {pw_cup/1_000:.0%}.")
print(f"3) Руками и OLS — одна оценка: {d_hand:+.5f} = {beta[1]:+.5f}; θ = коэф. при X.")
print("4) Пост-стратификация = CUPED со стратой-ковариатой; непрерывная пре-метрика сильнее.")
print(f"5) A/A-уровень CUPED = {np.mean(p_aa < ALPHA):.1%} — тест честен под H0.")
print(f"6) Пост-тритментная ковариата съела {eaten:.0%} эффекта — только пре-период!")
