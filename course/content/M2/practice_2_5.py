# -*- coding: utf-8 -*-
"""Практика 2.5 — ANOVA: мостик к регрессии и CUPED (сквозной кейс «ЕдаДома»).

Не верь — проверь симуляцией.

Что делаем:
1) однофакторный ANOVA руками на 4 вариантах пост-заказного экрана
   (метрика — время до второго заказа, дни): SS_between/SS_within, df,
   F, p + сверка со scipy.stats.f_oneway;
2) под H0 p-value ANOVA равномерен (хороший тест), а 6 попарных t-тестов
   без поправки красятся в ~1-(1-0.05)^6 = 26% миров — вот зачем омнибус;
3) F = t^2 для двух групп — численная проверка;
4) ANOVA = линейная регрессия с dummy-кодировкой: numpy.linalg.lstsq даёт
   те же F/p, коэффициенты = средние групп; бонус — добавляем ковариату
   пре-периода (задел на CUPED/ANCOVA, урок 3.5);
5) post-hoc: попарные t + Холм и Тьюки HSD (statsmodels).

Запуск:  python3 practice_2_5.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# # Практика 2.5 — ANOVA как регрессия
# Один тест на «хоть одна группа отличается» (омнибус) вместо россыпи
# попарных сравнений. А потом покажем, что ANOVA — это обычная линейная
# регрессия с dummy-предикторами: та самая оптика, из которой растут
# CUPED и ANCOVA (М3.5).

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

RNG = np.random.default_rng(15)  # seed -> результат воспроизводим
HERE = Path(__file__).resolve().parent
ALPHA = 0.05

# %% [markdown]
# ## Шаг 1. ANOVA руками: 4 варианта пост-заказного экрана
# A/B/n-тест: после доставки показывается один из 4 экранов «закажите снова».
# Вариант D — агрессивный апселл. Метрика — время до второго заказа (дни).

# %%
groups = {
    "A": np.array([10, 12, 14, 16, 18]),
    "B": np.array([12, 14, 16, 18, 20]),
    "C": np.array([11, 13, 15, 17, 19]),
    "D": np.array([18, 20, 22, 24, 26]),
}
k = len(groups)                     # число групп
n_per = 5
n = k * n_per
means = {g: v.mean() for g, v in groups.items()}
grand = np.concatenate(list(groups.values())).mean()

ss_b = sum(n_per * (means[g] - grand) ** 2 for g in groups)   # вариативность МЕЖДУ группами
ss_w = sum(((groups[g] - means[g]) ** 2).sum() for g in groups)  # вариативность ВНУТРИ групп
df_b, df_w = k - 1, n - k
ms_b, ms_w = ss_b / df_b, ss_w / df_w
F_hand = ms_b / ms_w
p_hand = stats.f.sf(F_hand, df_b, df_w)

F_scipy, p_scipy = stats.f_oneway(*groups.values())

anova_table = pd.DataFrame(
    {
        "SS": [ss_b, ss_w, ss_b + ss_w],
        "df": [df_b, df_w, df_b + df_w],
        "MS": [ms_b, ms_w, np.nan],
        "F": [F_hand, np.nan, np.nan],
        "p-value": [p_hand, np.nan, np.nan],
    },
    index=["между группами", "внутри групп (остаток)", "итого"],
)
print("=" * 80)
print("1) 4 варианта экрана, время до второго заказа (дни), n = 5 на группу")
print(f"   средние: " + ", ".join(f"{g} = {means[g]:.1f}" for g in groups) + f"; общее = {grand:.2f}")
print(anova_table.round(4).to_string())
print(f"   руки: F = {F_hand:.4f}, p = {p_hand:.4f}")
print(f"   scipy.stats.f_oneway: F = {F_scipy:.4f}, p = {p_scipy:.4f} -> совпало")
print(f"   эта-квадрат = SS_b/SS_total = {ss_b / (ss_b + ss_w):.2f} — 55% разброса объясняется группой")
print("   H0 «все средние равны» отвергнута: хотя бы один вариант отличается.")

# %%
fig, ax = plt.subplots(figsize=(8.2, 4.5))
colors = {"A": "#4C72B0", "B": "#55A868", "C": "#8172B3", "D": "#DD8452"}
for i, (g, v) in enumerate(groups.items()):
    ax.scatter(np.full(n_per, i) + RNG.uniform(-0.08, 0.08, n_per), v,
               s=55, color=colors[g], alpha=0.8, zorder=3, label=f"{g}: mean = {means[g]:.1f}")
    ax.hlines(means[g], i - 0.25, i + 0.25, color=colors[g], lw=3, zorder=4)
ax.hlines(grand, -0.4, k - 0.6, color="black", ls="--", lw=1.6, label=f"общее среднее = {grand:.2f}")
for i, g in enumerate(groups):
    ax.annotate("", xy=(i + 0.33, means[g]), xytext=(i + 0.33, grand),
                arrowprops=dict(arrowstyle="<->", color="#C44E52", lw=1.4))
ax.text(k - 0.55, 22.8, "SS_between:\nразброс средних", color="#C44E52", fontsize=9)
ax.text(0.35, 5.5, "SS_within:\nразброс внутри групп", color="gray", fontsize=9)
ax.set_xticks(range(k))
ax.set_xticklabels([f"{g}" for g in groups])
ax.set_ylabel("дни до второго заказа")
ax.set_title(f"ANOVA = «шум между» / «шум внутри»: F = {F_hand:.2f}, p = {p_hand:.4f}")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(HERE / "practice_2_5_anova.png", dpi=150)
print("Сохранено: practice_2_5_anova.png")

# %% [markdown]
# ## Шаг 2. Под H0 p-value ANOVA равномерен; попарные t без поправки — красятся
# 3000 миров без эффекта: 4 группы из одного распределения. Омнибус-ANOVA
# ведёт себя как честный тест (p<0.05 в ~5% миров), а 6 попарных t-тестов
# ловят «разницу» в каждом четвёртом мире.

# %%
WORLDS = 3_000
n_h0 = 20
p_omnibus, pairwise_fwer = [], 0
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
for _ in range(WORLDS):
    data = RNG.normal(15.0, 2.5, size=(k, n_h0))  # H0: одна популяция
    p_omnibus.append(stats.f_oneway(*data).pvalue)
    p_pairs = [stats.ttest_ind(data[i], data[j]).pvalue for i, j in pairs]
    pairwise_fwer += min(p_pairs) < ALPHA

p_omnibus = np.array(p_omnibus)
print("=" * 80)
print("2) Под H0 (эффектов нет), 4 группы по n=20, {} миров".format(WORLDS))
print(f"   ANOVA: доля p<0.05 = {np.mean(p_omnibus < ALPHA):.1%} (номинал 5%)")
print(f"   6 попарных t без поправки: FWER = {pairwise_fwer / WORLDS:.1%}"
      f" (теория 1-0.95^6 = {1 - 0.95 ** 6:.0%})")
print("   Вот зачем омнибус: одна гипотеза «все равны» вместо шести.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
axes[0].hist(p_omnibus, bins=30, color="#4C72B0", alpha=0.9)
axes[0].axhline(WORLDS / 30, color="#C44E52", ls="--", lw=1.6, label="уровень равномерности")
axes[0].set_title(f"p-value ANOVA под H0 равномерен\nдоля p<0.05 = {np.mean(p_omnibus < ALPHA):.1%}")
axes[0].set_xlabel("p-value")
axes[0].legend(fontsize=9)
axes[1].bar(["ANOVA (омнибус)", "6 попарных t\nбез поправки"],
            [np.mean(p_omnibus < ALPHA), pairwise_fwer / WORLDS],
            color=["#55A868", "#C44E52"], alpha=0.9, width=0.5)
axes[1].axhline(ALPHA, color="black", ls="--", lw=1.6)
axes[1].set_ylabel("доля миров с >=1 «открытием» под H0")
axes[1].set_title("Одна проверка против шести: FWER 5% против 26%")
fig.suptitle("Честный тест держит 5% под H0 — множественность ломает уровень",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "practice_2_5_puniform.png", dpi=150)
print("Сохранено: practice_2_5_puniform.png")

# %% [markdown]
# ## Шаг 3. F = t^2 при двух группах
# ANOVA с k=2 — это тот же самый t-тест: F-статистика равна квадрату t.

# %%
t2 = stats.ttest_ind(groups["A"], groups["B"], equal_var=True)
F2, p_F2 = stats.f_oneway(groups["A"], groups["B"])
print("=" * 80)
print("3) Две группы (A vs B):")
print(f"   t = {t2.statistic:.4f}, t^2 = {t2.statistic ** 2:.4f}, p = {t2.pvalue:.4f}")
print(f"   F = {F2:.4f}, p = {p_F2:.4f}  ->  F = t^2, p совпадают")

# %% [markdown]
# ## Шаг 4. ANOVA = регрессия с dummy-предикторами
# Дизайн-матрица X = [константа, B, C, D] (A — референсная группа).
# МНК-оценки — средние групп; F-тест регрессии — тот же F, что у ANOVA.
# А теперь трюк М3.5: добавим в X колонку «выручка в пре-периоде»
# (коррелирует с метрикой) — остаточная дисперсия упадёт, SE эффекта
# сожмётся. Это и есть CUPED/ANCOVA глазами линейной алгебры.

# %%
y = np.concatenate([groups[g] for g in groups])
labels = np.concatenate([[g] * n_per for g in groups])
d_b = (labels == "B").astype(float)
d_c = (labels == "C").astype(float)
d_d = (labels == "D").astype(float)
X = np.column_stack([np.ones(n), d_b, d_c, d_d])


def ols_f_test(Xm, yv):
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    y_hat = Xm @ beta
    resid = yv - y_hat
    sse = resid @ resid
    ssr = ((y_hat - yv.mean()) ** 2).sum()
    p_dim = Xm.shape[1] - 1  # число предикторов кроме константы
    df_err = len(yv) - Xm.shape[1]
    f_stat = (ssr / p_dim) / (sse / df_err)
    return beta, f_stat, stats.f.sf(f_stat, p_dim, df_err), sse / df_err


beta, F_reg, p_reg, mse_reg = ols_f_test(X, y)
print("=" * 80)
print("4) ANOVA = регрессия с dummy")
print(f"   коэффициенты МНК: константа = {beta[0]:.2f} (= среднее A), "
      f"B = {beta[1]:+.2f}, C = {beta[2]:+.2f}, D = {beta[3]:+.2f}")
print("   (средние групп: " + ", ".join(f"{g} = {means[g]:.1f}" for g in groups) + ")")
print(f"   F регрессии = {F_reg:.4f}, p = {p_reg:.4f} -> совпадает с ANOVA (шаг 1)")

# ковариата пре-периода: измерена ДО теста, поэтому коррелирует с метрикой
# только через внутриgroupовой шум (эффекта тритмента в ней нет — иначе каша)
RHO = 0.8
mu_vec = np.concatenate([np.full(n_per, means[g]) for g in groups])
eps = y - mu_vec                      # внутриgroupовой шум метрики
pre = RHO * eps / eps.std() + np.sqrt(1 - RHO ** 2) * RNG.standard_normal(n)
X_with_cov = np.column_stack([X, pre])
beta2, F_reg2, p_reg2, mse_reg2 = ols_f_test(X_with_cov, y)

# SE коэффициента при D (эффект варианта D) до/после ковариаты
def se_coef(Xm):
    beta_, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    resid = y - Xm @ beta_
    mse = (resid @ resid) / (len(y) - Xm.shape[1])
    return np.sqrt(np.diag(np.linalg.inv(Xm.T @ Xm)) * mse)

se_wo, se_wi = se_coef(X), se_coef(X_with_cov)
print(f"   добавили ковариату пре-периода (rho={RHO}): MS_остатка {mse_reg:.2f} -> {mse_reg2:.2f},")
print(f"   SE эффекта D: {se_wo[3]:.3f} -> {se_wi[3]:.3f} (сжатие в {se_wo[3] / se_wi[3]:.2f} раза,"
      f" теория ~1/sqrt(1-rho^2) = {1 / np.sqrt(1 - RHO ** 2):.2f})")
print(f"   p эффекта D: {2 * stats.t.sf(abs(beta[3] / se_wo[3]), n - 4):.4f} -> "
      f"{2 * stats.t.sf(abs(beta2[3] / se_wi[3]), n - 5):.4f}")
print("   Это CUPED/ANCOVA в миниатюре — подробно в уроке 3.5.")

# %% [markdown]
# ## Шаг 5. Post-hoc: где именно различия
# ANOVA сказал «не все равны». Кто от кого отличается? Попарные t
# (с общей дисперсией MS_w) + Холм; рядом — Тьюки HSD.

# %%
msw = ms_w
rows = []
for i, gi in enumerate(groups):
    for gj in list(groups)[i + 1:]:
        se = np.sqrt(msw * (2 / n_per))
        t_stat = (means[gi] - means[gj]) / se
        rows.append((f"{gi}–{gj}", means[gi] - means[gj], t_stat,
                     2 * stats.t.sf(abs(t_stat), df_w)))
tab = pd.DataFrame(rows, columns=["пара", "разность", "t", "p_raw"])
tab["p_holm"] = multipletests(tab["p_raw"], method="holm")[1]
tab["значимо (Холм)"] = tab["p_holm"] < ALPHA
print("=" * 80)
print("5) Post-hoc: попарные t + Холм")
print(tab.round(4).to_string(index=False))
print("   Вариант D отличается от A, B, C; A, B, C между собой — нет:")
print("   агрессивный апселл откладывает повторный заказ.")

from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(y, labels, alpha=ALPHA)
print("\n   Тьюки HSD (тот же FWER, свои пороги):")
print(tukey.summary())

# %%
print()
print("=" * 80)
print("ГЛАВНОЕ (что должны увидеть):")
print(f"1) ANOVA руками: SS_b = {ss_b:.2f}, SS_w = {ss_w:.2f}, F = {F_hand:.2f},")
print(f"   p = {p_hand:.4f} — совпало с f_oneway до знака.")
print(f"2) Под H0 p ANOVA равномерен ({np.mean(p_omnibus < ALPHA):.1%} < 0.05);")
print(f"   попарные t без поправки: FWER {pairwise_fwer / WORLDS:.0%} (теория 26%).")
print(f"3) k=2: F = t^2 ({t2.statistic ** 2:.3f} = {F2:.3f}) — ANOVA обобщает t-тест.")
print("4) ANOVA = регрессия на dummy: коэффициенты = средние, F/p те же;")
print(f"   ковариата пре-периода сжала SE эффекта D в {se_wo[3] / se_wi[3]:.2f} раза")
print("   — это CUPED/ANCOVA (М3.5), живущая в той же дизайн-матрице.")
print("5) Post-hoc нужен всегда: omnibus «не все равны» != «все разные»;")
print("   Холм на попарных и Тьюки здесь дали одни и те же выводы.")
