# -*- coding: utf-8 -*-
"""Практика 4.5 — Проблемы АБ в бизнесе: кумулятивные эффекты, холдауты, экономика
(сквозной кейс «ЕдаДома»).

Не верь — проверяй симуляцией.

Что делаем:
1) ГОД РЕЛИЗОВ (2000 миров): 10 фич с истинными эффектами и шумом теста;
   катим только «зелёные» (значимо положительные) -> winner's curse:
   средний ЗАЯВЛЕННЫЙ эффект зелёных против истинного у этих же фич;
2) ВЗАИМОДЕЙСТВИЕ ФИЧ: факториал 2x2 — A и B по +5%, взаимодействие -6%;
   по отдельности оба теста зелёные, а совместный (A+B против контроля) — серый;
   наивная сумма +10% против фактических +3.6%;
3) ХОЛДАУТ-СВЕРКА: 200 тыс. юзеров на уровне месяцев — 95% получают фичи
   последовательно (каждая пара каннибализирует 0.5% прироста), 5% (холдаут)
   не получают ничего; сезонность дрейфует одинаково у всех. Сравниваем
   «сумму релизов по отчётам» с приростом против глобального холдаута;
4) ЭКОНОМИКА ТЕСТА: сколько недель тестить — стоимость задержки растёт
   линейно, стоимость пропуска хорошей фичи падает с мощностью, а вероятность
   ложного роллаута от длительности НЕ зависит (её покупает альфа). Ищем
   оптимум и смотрим два сценария.

Запуск из корня репозитория: python3 course/modules/M4/practice/practice_4_5.py
Графики сохраняются в папку images/ этого модуля.
"""

# %% [markdown]
# # Практика 4.5 — год релизов, winner's curse, холдаут, экономика

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(45)  # seed -> результат воспроизводим
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05
Z = stats.norm.ppf(1 - ALPHA / 2)  # 1.96


def relative_lift_and_se(mean_t, var_t, n_t, mean_c, var_c, n_c):
    """Дельта-метод для независимых рук: μT/μC−1, включая шум baseline."""
    if mean_c <= 0 or min(n_t, n_c) < 2:
        raise ValueError("Нужны положительный baseline и минимум 2 юнита в каждой руке")
    ratio = mean_t / mean_c
    se = np.sqrt(var_t / n_t + ratio**2 * var_c / n_c) / mean_c
    return ratio - 1, se

# %% [markdown]
# ## Шаг 1. Год из 10 фич: отбор по значимости переоценивает победителей
#
# Истинные относительные эффекты 10 фич «ЕдаДома» (на выручку юзера).
# Каждый запуск — тест 60 000 юзеров на руку; метрика с тяжёлым хвостом
# (CV = 2.3) даёт SE ОТНОСИТЕЛЬНОГО аплифта ~1.3%. Катим только зелёные
# (p < 0.05 и аплифт > 0). Смотрим, что команда НАПИШЕТ в годовом отчёте.

# %%
TRUE_EFFECTS = np.array(
    [0.035, 0.025, 0.000, 0.045, -0.010, 0.030, 0.000, 0.020, 0.040, -0.020]
)
N_FEATURES = len(TRUE_EFFECTS)
WORLDS = 2000
CV = 2.3          # коэффициент вариации выручки юзера за окно теста
N_ARM = 60_000    # юзеров на руку в каждом запуске
SE_REL = CV * np.sqrt(2.0 / N_ARM)

power = stats.norm.cdf(np.abs(TRUE_EFFECTS) / SE_REL - Z)
plan = pd.DataFrame(
    {
        "фича": [f"F{i + 1}" for i in range(N_FEATURES)],
        "истинный_эффект_%": TRUE_EFFECTS * 100,
        "мощность_теста": np.round(power, 2),
    }
)
print("План года: истинные эффекты и мощность каждого запуска")
print(plan.round(3).to_string(index=False))
print(f"\nSE относительного аплифта одного теста: {SE_REL * 100:.2f} п.п.")

measured = TRUE_EFFECTS[None, :] + RNG.normal(0, SE_REL, size=(WORLDS, N_FEATURES))
pval = 2 * stats.norm.sf(np.abs(measured / SE_REL))
green = (pval < ALPHA) & (measured > 0)

true_bcast = np.broadcast_to(TRUE_EFFECTS, measured.shape)
sel_claim = measured[green].mean()      # средний ЗАЯВЛЕННЫЙ эффект зелёных
sel_true = true_bcast[green].mean()     # средний истинный эффект этих же фич
claim_sum = np.where(green, measured, 0.0).sum(axis=1)         # заявленный итог года
true_sum = np.where(green, TRUE_EFFECTS[None, :], 0.0).sum(axis=1)  # истинный итог

print(f"\n--- {WORLDS} симуляционных миров ---")
print(f"зелёных за год: в среднем {green.sum(axis=1).mean():.1f} из {N_FEATURES}")
print(f"средний заявленный эффект зелёных:      +{sel_claim * 100:.2f}%")
print(f"средний ИСТИННЫЙ эффект этих же фич:    +{sel_true * 100:.2f}%")
print(f"переоценка победителей (winner's curse): x{sel_claim / sel_true:.2f}")
print(f"итог года по отчётам: {claim_sum.mean() * 100:.1f}% заявлено"
      f" против {true_sum.mean() * 100:.1f}% истинно")
print(f"переоценка суммы года: +{(claim_sum - true_sum).mean() * 100:.1f} п.п.")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
ax = axes[0]
idx = np.arange(measured.size)[green.ravel()]
show = RNG.choice(idx, size=min(3000, idx.size), replace=False)
r, c = np.divmod(show, N_FEATURES)
ax.scatter(true_bcast[r, c] * 100, measured[r, c] * 100, s=9, alpha=0.35,
           color="#1f77b4", label="зелёные тесты (миры x фичи)")
lim = [-2, 8]
ax.plot(lim, lim, "k--", lw=1, label="измеренное = истинному")
ax.axhline(0, color="grey", lw=0.7)
ax.axvline(sel_true * 100, color="C3", ls=":", lw=1.5)
ax.axhline(sel_claim * 100, color="C3", lw=1.5)
ax.annotate(f"заявляем +{sel_claim * 100:.1f}%\nистина +{sel_true * 100:.1f}%\n"
            f"переоценка x{sel_claim / sel_true:.2f}",
            xy=(sel_true * 100, sel_claim * 100), xytext=(0.5, 6.3),
            fontsize=10, color="C3",
            arrowprops=dict(arrowstyle="->", color="C3"))
ax.set_xlabel("истинный эффект, %")
ax.set_ylabel("измеренный (заявленный) эффект, %")
ax.set_title("Точки зелёных тестов висят НАД диагональю")
ax.legend(loc="lower right", fontsize=8)

ax = axes[1]
diff = (claim_sum - true_sum) * 100
ax.hist(diff, bins=60, color="#1f77b4", alpha=0.8)
ax.axvline(diff.mean(), color="C3", lw=2)
ax.annotate(f"в среднем +{diff.mean():.1f} п.п. воздуха\nв годовом отчёте",
            xy=(diff.mean(), ax.get_ylim()[1] * 0.75), xytext=(diff.mean() + 1.5, ax.get_ylim()[1] * 0.85),
            fontsize=10, color="C3", arrowprops=dict(arrowstyle="->", color="C3"))
ax.set_xlabel("заявленная сумма минус истинная, п.п.")
ax.set_ylabel("число миров")
ax.set_title("Годовой отчёт систематически обещает больше, чем есть")
fig.suptitle("Winner's curse: отбор по значимости смещает эффекты вверх", y=1.02)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_4_5_winners_curse.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("-> practice_4_5_winners_curse.png")

# %% [markdown]
# ## Шаг 2. Две зелёные фичи с отрицательным взаимодействием
#
# F1 «умная сортировка каталога» (+5% конверсии в заказ) и F2 «бейдж Хит»
# (+5%). Тесты спроектированы под MDE +5% при 80% мощности. Но сортировка
# уже поднимает хиты наверх — бейдж дублирует сигнал: взаимодействие -6%.
# Каждый тест честно сравнивает «мир с фичой» против «мира без неё».
# А вместе их никто не тестировал.

# %%
P0, EFF_A, EFF_B, INTERACT = 0.10, 0.05, 0.05, -0.06
N2 = 56_500  # юзеров на руку: мощность ~80% на относительный эффект +5%
P_A = P0 * (1 + EFF_A)
P_B = P0 * (1 + EFF_B)
P_AB = P0 * (1 + EFF_A) * (1 + EFF_B) * (1 + INTERACT)

print(f"истинные конверсии: контроль {P0:.3f}, A {P_A:.4f}, B {P_B:.4f}, A+B {P_AB:.5f}")
print(f"наивная сумма: +{(1 + EFF_A) * (1 + EFF_B) * 100 - 100:.2f}%,"
      f" фактический A+B: +{P_AB / P0 * 100 - 100:.2f}%")


def prop_test(x_c, x_t, n):
    """Двухвыборочный z-тест долей + относительный аплифт с 95% ДИ."""
    p_c, p_t = x_c / n, x_t / n
    p_pool = (x_c + x_t) / (2 * n)
    se = np.sqrt(p_pool * (1 - p_pool) * 2 / n)
    zst = (p_t - p_c) / se
    pv = 2 * stats.norm.sf(abs(zst))
    uplift, se_rel = relative_lift_and_se(p_t, p_t * (1-p_t), n,
                                        p_c, p_c * (1-p_c), n)
    return uplift, pv, 1.96 * se_rel


DEMO = np.random.default_rng(4502)  # демонстрационный мир (типичный паттерн)
x_c = DEMO.binomial(N2, P0)
x_a = DEMO.binomial(N2, P_A)
x_b = DEMO.binomial(N2, P_B)
x_ab = DEMO.binomial(N2, P_AB)

rows = []
for name, x_t in [("A: сортировка", x_a), ("B: бейдж", x_b), ("A+B вместе", x_ab)]:
    up, pv, ci = prop_test(x_c, x_t, N2)
    rows.append({"тест": name, "аплифт_%": up * 100, "p_value": pv,
                 "вердикт": "ЗЕЛЁНЫЙ" if (pv < ALPHA and up > 0) else "серый"})
demo_tab = pd.DataFrame(rows)
print("\nДемонстрационный мир (реализованные конверсии одного мира):")
print(demo_tab.round(4).to_string(index=False))

# То же самое на 10 000 миров (нормальное приближение выборочных долей):
W2 = 10_000
rng2 = np.random.default_rng(21)
mc = P0 + rng2.normal(0, np.sqrt(P0 * (1 - P0) / N2), W2)
ma = P_A + rng2.normal(0, np.sqrt(P_A * (1 - P_A) / N2), W2)
mb = P_B + rng2.normal(0, np.sqrt(P_B * (1 - P_B) / N2), W2)
mab = P_AB + rng2.normal(0, np.sqrt(P_AB * (1 - P_AB) / N2), W2)


def green_rate(mt, mctrl, pt):
    up = mt / mctrl - 1
    se = np.sqrt(pt * (1 - pt) * 2 / N2) / P0
    pv = 2 * stats.norm.sf(np.abs(up / se))
    return ((pv < ALPHA) & (up > 0)).mean()


p_a_g = green_rate(ma, mc, P_A)
p_b_g = green_rate(mb, mc, P_B)
p_ab_g = green_rate(mab, mc, P_AB)
print(f"\nНа {W2} миров: A зелёный в {p_a_g * 100:.0f}%, B в {p_b_g * 100:.0f}%,"
      f" A+B в {p_ab_g * 100:.0f}% миров")
se_ab = np.sqrt(P_AB * (1 - P_AB) * 2 / N2) / P0
pv_ab = 2 * stats.norm.sf(np.abs((mab / mc - 1) / se_ab))
both_green_ab_grey = (
    (2 * stats.norm.sf(np.abs((ma / mc - 1) / (np.sqrt(P_A * (1 - P_A) * 2 / N2) / P0))) < ALPHA)
    & (ma / mc > 1)
    & (2 * stats.norm.sf(np.abs((mb / mc - 1) / (np.sqrt(P_B * (1 - P_B) * 2 / N2) / P0))) < ALPHA)
    & (mb / mc > 1)
    & (pv_ab >= ALPHA)
).mean()
print(f"сценарий «оба зелёные, вместе серый»: {both_green_ab_grey * 100:.0f}% миров")

fig, ax = plt.subplots(figsize=(9.5, 5.5))
labels = ["A:\nсортировка", "B:\nбейдж", "наивная\nсумма A+B", "фактически\nA+B (тест)"]
vals, errs, colors = [], [], []
up_a, _, ci_a = prop_test(x_c, x_a, N2)
up_b, _, ci_b = prop_test(x_c, x_b, N2)
up_ab, _, ci_ab = prop_test(x_c, x_ab, N2)
vals = [up_a * 100, up_b * 100, ((1 + EFF_A) * (1 + EFF_B) - 1) * 100, up_ab * 100]
errs = [ci_a * 100, ci_b * 100, 0.0, ci_ab * 100]
colors = ["#2ca02c", "#2ca02c", "#7f7f7f", "#1f77b4"]
bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=colors, alpha=0.85,
              error_kw=dict(lw=1.5))
bars[2].set_hatch("//")
bars[2].set_edgecolor("white")
ax.axhline((P_AB / P0 - 1) * 100, color="C3", ls="--", lw=1.5)
ax.annotate(f"истинный эффект A+B: +{(P_AB / P0 - 1) * 100:.1f}%",
            xy=(3, (P_AB / P0 - 1) * 100), xytext=(1.1, 8.6), color="C3", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="C3"))
for i, (v, pv) in enumerate(zip(vals, [prop_test(x_c, x_a, N2)[1],
                                       prop_test(x_c, x_b, N2)[1],
                                       np.nan, prop_test(x_c, x_ab, N2)[1]])):
    tag = "зелёный" if pv < ALPHA else ("серый" if not np.isnan(pv) else "никто не тестировал")
    ax.text(i, v + (errs[i] if errs[i] else 0.15) + 0.15, tag, ha="center", fontsize=9)
ax.set_ylabel("относительный прирост конверсии, %")
ax.set_title("Сумма зелёных +10%, а вместе — +3.6% и серый тест")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_4_5_interaction.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("-> practice_4_5_interaction.png")

# %% [markdown]
# ## Шаг 3. Холдаут-сверка: сумма релизов против глобального холдаута
#
# 200 000 юзеров: 95% — «прод», куда раз в месяц выкатывается фича (только
# зелёные по итогам launch-теста), 5% — холдаут, не получают НИЧЕГО. Каждая
# пара живых фич каннибализирует 0.5% суммарного прироста (внимание юзера
# не резиновое). Сезонность (спад в январе, пик летом) действует на всех
# одинаково — рандомизация защищает сравнение от дрейфа.

# %%
N_USERS, HOLD_N, MONTHS = 200_000, 10_000, 12
LAMBDA_PAIR = 0.005
SEASON = np.array([0.97, 0.96, 1.00, 1.02, 1.06, 1.08,
                   1.12, 1.10, 1.05, 1.02, 0.99, 1.05])

# персистентная «тяжесть» юзера (кто заказывает часто, тот и в следующем месяце)
SIG_U = 0.8
u = RNG.lognormal(-SIG_U**2 / 2, SIG_U, size=N_USERS)
is_hold = np.zeros(N_USERS, dtype=bool)
is_hold[:HOLD_N] = True

# месячная выручка юзера: логнормальная, среднее 433 руб, тяжёлый хвост
MU_M, CV_M = 433.3, 2.3
sig_log = np.sqrt(np.log(1 + CV_M**2) - SIG_U**2)
mu_log = np.log(MU_M) - sig_log**2 / 2

def feature_multiplier(effects, penalty=LAMBDA_PAIR):
    """Один outcome model для launch-контрастов и глобального холдаута."""
    k = len(effects)
    interaction = 1 - penalty*k*(k-1)/2
    if interaction <= 0:
        raise ValueError("Модель взаимодействий требует положительного множителя")
    return np.prod(1 + np.asarray(effects, dtype=float))*interaction


launched = []            # индексы раскатанных (зелёных) фич
launch_log = []
claim_meas = {}          # индекс фичи -> заявленный аплифт
for i in range(N_FEATURES):          # фичи F1..F10 запускаются в месяцы 1..10
    prior = len(launched)
    current = TRUE_EFFECTS[launched]
    true_marg = feature_multiplier(np.append(current, TRUE_EFFECTS[i])) / feature_multiplier(current) - 1
    meas = true_marg + RNG.normal(0, SE_REL)
    pv = 2 * stats.norm.sf(abs(meas / SE_REL))
    ok = (pv < ALPHA) and (meas > 0)
    launch_log.append((f"F{i + 1}", TRUE_EFFECTS[i] * 100, true_marg * 100,
                       meas * 100, pv, "катим" if ok else "не катим"))
    if ok:
        launched.append(i)
        claim_meas[i] = meas

launch_tab = pd.DataFrame(
    launch_log, columns=["фича", "истинный_%", "маргинальный_%", "заявленный_%",
                         "p_value", "решение"]
)
print("Launch-тесты года (маргинальный эффект — против ТЕКУЩЕГО прода):")
print(launch_tab.round(4).to_string(index=False))

# фактический мультипликатор прода по месяцам: произведение эффектов зелёных
# минус каннибализация пар
k_live = np.zeros(MONTHS, dtype=int)
M = np.ones(MONTHS)
for i in launched:
    M[i:] *= 1 + TRUE_EFFECTS[i]
    k_live[i:] += 1
pairs = np.array([LAMBDA_PAIR * k * (k - 1) / 2 for k in k_live])
mult_treat = M * (1 - pairs) * SEASON
mult_hold = SEASON

# юзер-уровень: месячная выручка = тяжесть юзера x логнормальный шум x мультипликатор
base = RNG.lognormal(mu_log, sig_log, size=(N_USERS, MONTHS))
rev = u[:, None] * np.where(is_hold[:, None], base * mult_hold[None, :], base * mult_treat[None, :])

treat_mask = ~is_hold
months = np.arange(1, MONTHS + 1)
claim_cum, gap, gap_lo, gap_hi = [], [], [], []
for m in range(MONTHS):
    claim_cum.append((np.prod([1 + claim_meas[i] for i in claim_meas if i <= m]) - 1) * 100)
    mt = rev[treat_mask, m].mean()
    mh = rev[is_hold, m].mean()
    g, se = relative_lift_and_se(mt, rev[treat_mask, m].var(ddof=1), treat_mask.sum(),
                                 mh, rev[is_hold, m].var(ddof=1), HOLD_N)
    gap.append(g * 100)
    gap_lo.append((g - 1.96 * se) * 100)
    gap_hi.append((g + 1.96 * se) * 100)

hold_tab = pd.DataFrame({
    "месяц": months, "фич_в_проде": k_live,
    "заявлено_накопл_%": np.round(claim_cum, 1),
    "холдаут_%": np.round(gap, 1),
    "холдаут_CI-": np.round(gap_lo, 1),
    "холдаут_CI+": np.round(gap_hi, 1),
})
print("\nПомесячная сверка (заявлено по отчётам против глобального холдаута):")
print(hold_tab.to_string(index=False))

ann_t = rev[treat_mask].sum(axis=1)
ann_h = rev[is_hold].sum(axis=1)
mt, mh = ann_t.mean(), ann_h.mean()
g, se = relative_lift_and_se(mt, ann_t.var(ddof=1), treat_mask.sum(),
                            mh, ann_h.var(ddof=1), HOLD_N)
ttest = stats.ttest_ind(ann_t, ann_h, equal_var=False)
naive_sum = (np.prod([1 + TRUE_EFFECTS[i] for i in launched]) - 1) * 100
claimed = sum(claim_meas.values()) * 100

print("\n--- Итоги года: три числа ---")
print(f"произведение отдельных истинных эффектов без взаимодействий: +{naive_sum:.1f}%")
print(f"сумма заявленного (описательная, не композиция):     +{claimed:.1f}%")
print(f"факт по глобальному холдауту:                 +{g * 100:.1f}%"
      f"  (95% ДИ относительного lift: {(g - 1.96 * se) * 100:+.1f}%..{(g + 1.96 * se) * 100:+.1f}%)")
print(f"t-тест Уэлча по годовой выручке: p = {ttest.pvalue:.2e}")
print(f"уровень декабря (все фичи в проду):        +{gap[-1]:.1f}%"
      f"  (95% ДИ lift: {gap_lo[-1]:+.1f}%..{gap_hi[-1]:+.1f}%)")
print(f"различие оценок разных горизонтов: {claimed - gap[-1]:.1f} п.п. из {claimed:.1f} заявленных"
      f" (разные окна экспозиции + шум + отбор; это не причинное разложение)")
print(f"дрейф сезонности за год: x{SEASON.max():.2f}..x{SEASON.min():.2f} — на сравнение"
      f" с холдаутом не повлиял (рандомизация)")

fig, ax = plt.subplots(figsize=(10.5, 5.5))
ax.step(months, claim_cum, where="post", color="#7f7f7f", lw=2,
        label="композиция оценённых последовательных lift")
ax.plot(months, naive_sum * np.ones(MONTHS), ":", color="#7f7f7f", lw=1.5,
        label=f"отдельные истинные эффекты без взаимодействий: +{naive_sum:.1f}%")
ax.plot(months, gap, "o-", color="#1f77b4", lw=2, label="факт: прирост против холдаута")
ax.fill_between(months, gap_lo, gap_hi, color="#1f77b4", alpha=0.2,
                label="точечные 95% ДИ холдаут-оценки")
ax.annotate(f"разрыв {claim_cum[-1] - gap[-1]:.1f} п.п.:\nотбор и шум оценок",
            xy=(12, gap[-1]), xytext=(7.2, claim_cum[-1] - 1.2), color="C3", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="C3"))
ax.set_xlabel("месяц года")
ax.set_ylabel("накопленный относительный прирост, %")
ax.set_title("Холдаут несмещён, но шумен: первые месяцы молчит даже при 10 тыс. юзеров")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_4_5_holdout.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("-> practice_4_5_holdout.png")

# %% [markdown]
# ## Шаг 4. Экономика: сколько недель тестить одну фичу
#
# Пуш об акциях: ожидаемый эффект +0.5% недельной выручки (80 млн -> ценность
# 400 тыс./нед). Приор «фича вообще работает» 40% (по Kohavi лишь ~1/3 идей
# выигрывают). Если работает — катим и получаем ценность; если нет, а тест
# ложно зелёный — отдельные эксплуатационные затраты 240 тыс./нед, откатим через 13 недель.
# У плохой фичи эффект OEC равен нулю; отрицательный эффект — иная альтернатива.
# Горизонт решения — 26 недель от сегодняшнего дня. Трафик 150 тыс. юзеров/нед на руку.
#
# - задержка(w) = w недель без фичи, если она работает: w * ценность * приор;
# - пропуск(w) = серый тест хорошей фичи -> не катим: (1-мощность) * ценность * (горизонт−недели теста);
# - ложный роллаут(w) = P(значимо зелёный | эффекта нет) * вред * недели до отката
#   — НЕ зависит от w (её снижает альфа, а не длительность!).

# %%
WEEKLY_REV, LIFT = 80e6, 0.005
VALUE_WK = WEEKLY_REV * LIFT                 # 400 тыс. руб/нед
PRIOR_WORKS = 0.40
HARM_WK = WEEKLY_REV * 0.003                 # 240 тыс. руб/нед
ROLLBACK_WEEKS, LIFE_WEEKS = 13, 26
USERS_WK_ARM, MEAN_WK, SD_WK = 150_000, 560.0, 500.0

weeks = np.arange(1, 9)
se_w = SD_WK * np.sqrt(2.0 / (USERS_WK_ARM * weeks))
power_w = stats.norm.cdf(LIFT * MEAN_WK / se_w - Z)
delay = weeks * VALUE_WK * PRIOR_WORKS
miss = (1 - power_w) * PRIOR_WORKS * VALUE_WK * (LIFE_WEEKS - weeks)
false_roll = (1 - PRIOR_WORKS) * (ALPHA / 2) * HARM_WK * ROLLBACK_WEEKS * np.ones_like(weeks)
total = delay + miss + false_roll

econ = pd.DataFrame({
    "недель_теста": weeks,
    "мощность": np.round(power_w, 3),
    "стоимость_задержки_тыс": np.round(delay / 1e3, 0),
    "стоимость_пропуска_тыс": np.round(miss / 1e3, 0),
    "ожидаемый_ложный_роллаут_тыс": np.round(false_roll / 1e3, 0),
    "итого_тыс": np.round(total / 1e3, 0),
})
print("Экономика теста push-фичи (тыс. руб ожидаемых потерь):")
print(econ.to_string(index=False))
w_opt = weeks[np.argmin(total)]
print(f"\nоптимум: {w_opt} недель (дальше задержка дорожает быстрее,"
      f" чем падает риск пропуска)")

# сценарий Б: мелкая фича — ценность 40 тыс/нед, срок жизни 8 недель, приор 30%
VALUE2, PRIOR2, LIFE2 = 40e3, 0.30, 8
total2 = (weeks * VALUE2 * PRIOR2
          + (1 - power_w) * PRIOR2 * VALUE2 * np.maximum(LIFE2 - weeks, 0)
          + (1 - PRIOR2) * (ALPHA / 2) * HARM_WK * ROLLBACK_WEEKS)
w_opt2 = weeks[np.argmin(total2)]
print(f"мелкая фича (40 тыс/нед, живёт 8 недель): оптимум {w_opt2} недели —"
      f" короткий тест выгоднее длинного")

# альфа-обмен: чтобы сохранить мощность при alpha=0.005, тест длиннее в
z_new = stats.norm.ppf(1 - 0.005 / 2)
factor = ((z_new + 0.84) / (Z + 0.84)) ** 2
print(f"альфа 0.05 -> 0.005 снижает вероятность ложного роллаута с 2.5% до 0.25%,"
      f" но ради той же мощности тест дольше в {factor:.1f} раза")

fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.plot(weeks, delay / 1e3, "o-", label="стоимость задержки (растёт линейно)")
ax.plot(weeks, miss / 1e3, "o-", label="стоимость пропуска хорошей фичи (падает с мощностью)")
ax.plot(weeks, false_roll / 1e3, "o-", label="ожидаемый ложный роллаут (не зависит от срока!)")
ax.plot(weeks, total / 1e3, "o-", lw=2.5, color="k", label="суммарные ожидаемые потери")
ax.axvline(w_opt, color="C3", ls="--", lw=1.5)
ax.annotate(f"оптимум: {w_opt} нед", xy=(w_opt, total[np.argmin(total)] / 1e3),
            xytext=(w_opt + 0.4, total[np.argmin(total)] / 1e3 + 900), color="C3",
            arrowprops=dict(arrowstyle="->", color="C3"))
ax.set_xlabel("длительность теста, недель")
ax.set_ylabel("ожидаемые потери, тыс. руб")
ax.set_title("При точечном нуле риск ошибки постоянен; при вредной альтернативе падает с n")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_4_5_economics.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("-> practice_4_5_economics.png")

print("\nГОТОВО: 4 графика в", IMAGE_DIR)
