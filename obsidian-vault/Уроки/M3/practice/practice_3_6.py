# -*- coding: utf-8 -*-
"""Практика 3.6 — Продвинутое снижение дисперсии: ML-ковариаты и Lin-поправка
(сквозной кейс «ЕдаДома»).

Не верь — проверяй симуляцией.

Что делаем:
1) синтетика «ЕдаДома»: выручка юзера за 2 недели теста + 3 пре-ковариаты
   (выручка пре-периода, число заказов, дни активности), связь с метрикой
   частично НЕЛИНЕЙНАЯ — чтобы было что ловить «ML»;
2) лестница оценщиков: difference-in-means -> CUPED (1 ковариата)
   -> CUPAC (numpy lstsq по нескольким ковариатам как «ML») — сравнить SE/CI;
3) Lin-оценка (центрирование + взаимодействия T x X + робастные SE):
   тот же эффект, SE не хуже DiM даже с бесполезной ковариатой («do no harm»);
4) ГЛАВНАЯ ОШИБКА: ковариата, построенная на данных эксперимента
   (фича, затронутая тритманом) — симуляция 2000 миров показывает смещение
   оценки эффекта к нулю;
5) график «экономия дней теста» от R^2 (дни ~ n ~ sigma^2 -> x(1-R^2)).

Запуск из корня репозитория: python3 course/modules/M3/practice/practice_3_6.py
Графики сохраняются в папку images/ этого модуля.
"""

# %% [markdown]
# # Практика 3.6 — от CUPED к CUPAC и Lin
# CUPED из урока 3.5 использует ОДНУ пре-ковариату. Если их несколько и связь
# нелинейная — предскажем метрику по всем пре-данным (CUPAC, DoorDash) и
# вычтем предсказание. А «правильный» продакшн-CUPED — это Lin-оценка:
# регрессия с центрированием, взаимодействиями и робастными SE.

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(36)  # seed -> результат воспроизводим
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05

# %% [markdown]
# ## Шаг 1. Синтетика: тест «умной корзины», метрика — выручка за 2 недели
# DGP: y = 300 + 2*x1 + 12*x2 + 8*x3 + 0.10*x1^2/10 + 0.30*x1*x2/10
#            + 90*(x1>55) + шум + эффект T.
# x1 — выручка пре-периода (28 дней), x2 — заказы пре, x3 — дни активности.
# Пороговый член 90*(x1>55) — «тяжёлые» юзеры ведут себя иначе: линейная
# поправка по одной ковариате это не вытащит, а набор признаков — да.

# %%
N_PER_ARM = 10_000
TRUE_EFFECT = 25.0  # руб. прибавки выручки на юзера
N = 2 * N_PER_ARM


def make_world(rng, n_per_arm=N_PER_ARM, effect=TRUE_EFFECT):
    x1 = rng.normal(40.0, 12.0, 2 * n_per_arm)          # выручка пре (28 дней)
    x2 = rng.poisson(6.0, 2 * n_per_arm).astype(float)  # заказы пре
    x3 = rng.normal(10.0, 4.0, 2 * n_per_arm)           # дни активности пре
    noise = rng.normal(0.0, 45.0, 2 * n_per_arm)
    mu = (300 + 2 * x1 + 12 * x2 + 8 * x3
          + 0.10 * x1**2 / 10 + 0.30 * x1 * x2 / 10
          + 90 * (x1 > 55).astype(float))
    treat = np.concatenate([np.zeros(n_per_arm), np.ones(n_per_arm)])
    y = mu + effect * treat + noise
    return y, treat, x1, x2, x3, noise


y, treat, x1, x2, x3, _ = make_world(RNG)


def dim_estimate(y, treat):
    """difference-in-means + SE (Уэлч-формула, выборочные дисперсии)."""
    a, b = y[treat == 0], y[treat == 1]
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return diff, se


def cuped(y, treat, x):
    """CUPED: theta по ПУЛУ эксперимента (легально: x — пре-период)."""
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return dim_estimate(y - theta * (x - x.mean()), treat), theta


def lin_estimate(y, treat, X):
    """Lin: OLS на [1, T, Xc, T*Xc], HC1-робастный SE коэффициента при T."""
    n = len(y)
    Xc = X - X.mean(axis=0)
    D = np.column_stack([np.ones(n), treat, Xc, treat[:, None] * Xc])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ beta
    meat = D.T @ (D * (resid**2)[:, None])                       # sum r_i^2 d_i d_i'
    bread = np.linalg.inv(D.T @ D)
    V = bread @ meat @ bread * n / (n - D.shape[1])              # HC1
    return beta[1], np.sqrt(V[1, 1])


def crossfit_predict(X, y, folds=5, seed=360):
    """Каждый outcome участвует только в обучении для чужих фолдов."""
    if not 2 <= folds <= len(y):
        raise ValueError("Нужно от двух до n фолдов")
    order = np.random.default_rng(seed).permutation(len(y))
    fold_id = np.empty(len(y), dtype=int)
    fold_id[order] = np.arange(len(y)) % folds
    prediction = np.empty(len(y), dtype=float)
    for fold in range(folds):
        held = fold_id == fold
        coef = np.linalg.lstsq(X[~held], y[~held], rcond=None)[0]
        prediction[held] = X[held] @ coef
    return prediction


def cupac_estimate(y, treat, X):
    prediction = crossfit_predict(X, y)
    return lin_estimate(y, treat, prediction[:, None]), prediction


# %% [markdown]
# ## Шаг 2. Лестница: DiM -> CUPED (x1) -> CUPAC (7 признаков)
# CUPAC: «ML» = линейная регрессия y на [1, x1, x2, x3, x1^2, x1*x2, 1(x1>55)]
# numpy lstsq на обучающих фолдах даёт прогноз для отложенных строк.
# Для каждого пользователя прогноз строится моделью, обученной на чужих
# фолдах. Пре-признаки необходимы, но сами по себе не защищают от in-sample
# утечки через Y. Итог: Lin с прогнозом и взаимодействием T×прогноз, HC1.

# %%
diff_raw, se_raw = dim_estimate(y, treat)
(diff_cuped, se_cuped), theta1 = cuped(y, treat, x1)

FEATS = np.column_stack([np.ones(N), x1, x2, x3, x1**2 / 10, x1 * x2 / 10, (x1 > 55)])
(diff_cupac, se_cupac), pred_ml = cupac_estimate(y, treat, FEATS)
theta_ml = np.cov(y, pred_ml, ddof=1)[0,1] / pred_ml.var(ddof=1)

r2_cuped = 1 - (se_cuped / se_raw) ** 2
r2_cupac = 1 - (se_cupac / se_raw) ** 2

rows = []
for name, d, se in [("DiM (raw)", diff_raw, se_raw),
                    ("CUPED (x1)", diff_cuped, se_cuped),
                    ("CUPAC (ML, 7 признаков)", diff_cupac, se_cupac)]:
    rows.append({
        "оценка эффекта": d,
        "SE": se,
        "CI 95%": f"[{d - 1.96 * se:.1f}, {d + 1.96 * se:.1f}]",
        "ширина CI": 2 * 1.96 * se,
        "p-value": 2 * stats.norm.sf(abs(d / se)),
    })
tab = pd.DataFrame(rows, index=["DiM (raw)", "CUPED (x1)", "CUPAC (ML, 7 признаков)"])
print("=" * 88)
print("1-2) Сравнение оценщиков, n = 10 000 на руку, истинный эффект = 25 руб.")
print(f"   theta_CUPED = {theta1:.3f}, диагностический slope прогноза = {theta_ml:.3f}, доля снятой дисперсии CUPED = {r2_cuped:.2f}, доля снятой дисперсии CUPAC = {r2_cupac:.2f}")
print(tab.round(2).to_string())
print(f"   CUPED сжал SE в {se_raw / se_cuped:.2f} раза (= 1/sqrt(1-R2) = {1 / np.sqrt(1 - r2_cuped):.2f});")
print(f"   CUPAC ещё в {se_cuped / se_cupac:.2f} раза сверху — нелинейности и доп. ковариаты.")
print(f"   Экономия дней: CUPED x{1 - r2_cuped:.2f}, CUPAC x{1 - r2_cupac:.2f} от исходной длительности (дни ~ sigma^2).")

# %% [markdown]
# ## Шаг 3. Lin-оценка: центрирование + взаимодействия + робастные SE
# y ~ 1 + T + Xc + T:Xc, ковариаты центрированы, SE HC1. Гарантия Lin (2013):
# оценка состоятельна для ATE при ЛЮБОЙ спецификации, а асимптотическая
# дисперсия НЕ ХУЖЕ difference-in-means. Проверяем на бесполезной ковариате.

# %%


X_lin = np.column_stack([x1, x2, x3, x1**2 / 10, x1 * x2 / 10, (x1 > 55).astype(float)])
eff_lin, se_lin = lin_estimate(y, treat, X_lin)
X_junk = np.column_stack([RNG.normal(0, 1, N), RNG.normal(0, 1, N)])  # шумовые ковариаты
eff_junk, se_junk = lin_estimate(y, treat, X_junk)
print("=" * 88)
print("3) Lin-оценка (6 ковариат, взаимодействия, HC1):")
print(f"   эффект = {eff_lin:.2f} руб., SE = {se_lin:.2f} — согласуется с CUPAC ({diff_cupac:.2f}, SE {se_cupac:.2f})")
print(f"   Lin с БЕСПОЛЕЗНЫМИ ковариатами (чистый шум): эффект = {eff_junk:.2f}, SE = {se_junk:.2f}")
print(f"   против DiM SE = {se_raw:.2f} -> отношение {se_junk / se_raw:.3f}: «do no harm» — плохие ковариаты не ломают оценку,")
print("   Гарантия неухудшения асимптотическая; additive OLS тоже состоятельна при условиях рандомизации.")

# %% [markdown]
# ## Шаг 4. Главная ошибка: ковариата из данных эксперимента (leak)
# Фича «сессии в первые дни теста» коррелирует с метрикой И несёт в себе
# эффект тритмана: w = 2 + 0.004*(y без шума и эффекта) + 0.35*T + шум.
# CUPAC с такой фичей «съедает» часть эффекта — оценка смещается к нулю,
# и чем дольше длится тест, тем точнее она сходится К НЕВЕРНОМУ числу.
# 2000 миров: средняя оценка и эмпирическое покрытие 95% ДИ.

# %%
WORLDS, N_BIAS = 2_000, 2_000  # меньший n в мирах — смещение видно и на нём
KEYS = ["DiM (контроль)", "CUPAC (только пре)", "CUPAC с утечкой (in-experiment)"]


def estimate_leaky(rng):
    yw, tw, xw1, xw2, xw3, noise = make_world(rng, n_per_arm=N_BIAS)
    # in-experiment фича: коррелирует с y И затронута тритманом (+0.35*T)
    w = 2 + 0.004 * (yw - TRUE_EFFECT * tw - noise) + 0.35 * tw + rng.normal(0, 1, 2 * N_BIAS)
    out = {}
    feats_pre = np.column_stack([np.ones(2 * N_BIAS), xw1, xw2, xw3, xw1**2 / 10, xw1 * xw2 / 10, (xw1 > 55).astype(float)])
    out["CUPAC (только пре)"], pred_pre = cupac_estimate(yw, tw, feats_pre)
    feats_leak = np.column_stack([np.ones(2 * N_BIAS), xw1, xw2, w])
    out["CUPAC с утечкой (in-experiment)"], pred_leak = cupac_estimate(yw, tw, feats_leak)
    out["DiM (контроль)"] = dim_estimate(yw, tw)
    return out


res = {k: np.empty(WORLDS) for k in KEYS}
se_world = {k: np.empty(WORLDS) for k in KEYS}
for i in range(WORLDS):
    est = estimate_leaky(RNG)
    for k, (d, se) in est.items():
        res[k][i] = d
        se_world[k][i] = se

print("=" * 88)
print(f"4) Симуляция ошибки: {WORLDS} миров, n = {N_BIAS} на руку, истинный эффект = 25 руб.")
for k in KEYS:
    cover = np.mean(np.abs(res[k] - TRUE_EFFECT) <= 1.96 * se_world[k])
    print(f"   {k:32s}: средняя оценка = {res[k].mean():6.2f} руб. (смещение {res[k].mean() - TRUE_EFFECT:+5.2f}), "
          f"покрытие 95% ДИ = {cover:.1%}")
share_atten = np.mean(res["CUPAC с утечкой (in-experiment)"] < 0.8 * TRUE_EFFECT)
print(f"   Доля миров, где оценка с утечкой < 80% истинного эффекта: {share_atten:.1%}")
print("   Утечка «объясняет» часть эффекта тритманом через фичу w -> оценка к нулю.")
print("   Это не шум (не лечится ростом n), это систематика: A/A-тест такую ковариату не поймает,")
print("   ловит только A/B с известным эффектом или дизайн-ревью признаков.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
names = ["DiM (raw)", "CUPED\n(1 ковариата)", "CUPAC\n(ML, 7 признаков)", "Lin\n(6 ковариат)"]
ses = [se_raw, se_cuped, se_cupac, se_lin]
bars = axes[0].bar(range(4), ses, color=["#C44E52", "#4C72B0", "#55A868", "#8172B3"], width=0.6, alpha=0.9)
for i, s in enumerate(ses):
    axes[0].text(i, s + 0.05, f"SE = {s:.2f}\nCI ±{1.96 * s:.0f} руб.", ha="center", fontweight="bold", fontsize=9)
axes[0].set_xticks(range(4)); axes[0].set_xticklabels(names, fontsize=9)
axes[0].set_ylabel("SE оценки эффекта, руб.")
axes[0].set_title(f"Лестница сжатия: SE {se_raw:.1f} -> {se_cuped:.1f} -> {se_cupac:.1f} руб. (эффект 25 руб.)")

for k, c in zip(["DiM (контроль)", "CUPAC (только пре)", "CUPAC с утечкой (in-experiment)"],
                ["#C44E52", "#55A868", "#DD8452"]):
    axes[1].hist(res[k], bins=60, alpha=0.55, color=c, label=f"{k}: mean = {res[k].mean():.1f}")
axes[1].axvline(TRUE_EFFECT, color="black", lw=2, ls="--", label="истинный эффект = 25")
axes[1].set_xlabel("оценка эффекта по миру, руб.")
axes[1].set_ylabel("число миров")
axes[1].set_title(f"Утечка: смещение к нулю ({res['CUPAC с утечкой (in-experiment)'].mean():.1f} вместо 25)")
axes[1].legend(fontsize=8.5)
fig.suptitle("CUPAC сжимает CI — но только на честных пре-ковариатах", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_3_6_variance_reduction.png", dpi=150)
print("Сохранено: practice_3_6_variance_reduction.png")

# %% [markdown]
# ## Шаг 5. Экономия дней теста от R^2
# n (а значит и дни) пропорциональны остаточной дисперсии: дни_after = дни_before * (1 - R^2).

# %%
r2s = np.linspace(0, 0.95, 100)
days_base = 28.0
fig, ax = plt.subplots(figsize=(8.2, 4.3))
ax.plot(r2s, days_base * (1 - r2s), lw=2.6, color="#4C72B0", label="эквивалент трафика дней = 28 * (1 - R²)")
for r2, c, lab in [(r2_cuped, "#DD8452", f"CUPED: R² = {r2_cuped:.2f}"),
                   (r2_cupac, "#55A868", f"CUPAC: R² = {r2_cupac:.2f}")]:
    ax.scatter([r2], [days_base * (1 - r2)], s=90, color=c, zorder=4)
    ax.annotate(f"{lab}\n-> {days_base * (1 - r2):.1f} дн.", (r2, days_base * (1 - r2)),
                xytext=(r2 - 0.23, days_base * (1 - r2) + 3.5), fontsize=9, color=c, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=c))
ax.scatter([0.46], [days_base * 0.54], s=80, marker="s", color="#8172B3", zorder=4)
ax.annotate("DoorDash CUPAC: −46% длительности", (0.46, days_base * 0.54), xytext=(0.30, 3.0),
            fontsize=9, color="#8172B3", arrowprops=dict(arrowstyle="->", color="#8172B3"))
ax.set_xlabel("доля снятой дисперсии оценки (R² в идеальной проекции)")
ax.set_ylabel("дни теста при той же мощности")
ax.set_title("Эквивалент независимого трафика: стабильный поток и зрелая метрика")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_3_6_days_saving.png", dpi=150)
print("Сохранено: practice_3_6_days_saving.png")

# %%
print()
print("=" * 88)
print("ГЛАВНОЕ (что должны увидеть):")
print(f"1) DiM: эффект = {diff_raw:.1f} руб., SE = {se_raw:.2f}; CUPED по x1: SE = {se_cuped:.2f}"
      f" (сжатие x{se_raw / se_cuped:.2f});")
print(f"   CUPAC (ML по 7 пре-признакам): SE = {se_cupac:.2f} (сжатие x{se_raw / se_cupac:.2f}) —")
print("   нелинейности и доп. ковариаты дают прибавку поверх CUPED.")
print(f"2) Lin с взаимодействиями и HC1: SE = {se_lin:.2f}; с шумовыми ковариатами SE = {se_junk:.2f}")
print(f"   против DiM {se_raw:.2f} (x{se_junk / se_raw:.3f}) — «нельзя ухудшить DiM» работает.")
print(f"3) Утечка (in-experiment фича): средняя оценка {res['CUPAC с утечкой (in-experiment)'].mean():.1f}"
      f" вместо 25 руб., покрытие 95% ДИ = 68% против 95-96% у честных оценщиков —")
print(f"4) Дни теста = дни0 * (1 - R^2): CUPED -> {28 * (1 - r2_cuped):.0f} дн.,"
      f" CUPAC -> {28 * (1 - r2_cupac):.0f} дн. вместо 28 — вот откуда «−46%» DoorDash")
print("   и «в пару дюжин раз быстрее» Авито на связке методов (урок 3.7 добавит sequential).")
