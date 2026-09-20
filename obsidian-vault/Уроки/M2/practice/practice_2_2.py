# -*- coding: utf-8 -*-
"""Практика 2.2 — t-тесты и условия применимости (кейс «ЕдаДома»).

Принцип курса: не верь названию теста — проверь его фактическую alpha
симуляцией (метод ГМ: генерируем H0-мир и сравниваем фактический FPR
с номиналом).

Что делаем:
1) (а) равные дисперсии и равные n: Стьюдент (pooled) и Уэлч оба держат
      alpha = 5%;
2) (б) неравные дисперсии + разные n: эмпирическая alpha у Стьюдента
      уезжает в разы, Уэлч держит номинал; обратная конфигурация — Стьюдент
      «тихо» консервативен (тоже сломан);
3) (в) парный vs непарный t-тест на одной задаче (NPS курьеров до/после):
      SD разностей против SD групп; сколько наблюдений сэкономил парный
      дизайн;
4) (г) скошенный логнормальный чек при малых n: эмпирическая alpha t-теста
      плавает — правило Денга ~100·s^2 из урока 1.3.

Запуск из корня репозитория: python3 course/modules/M2/practice/practice_2_2.py
Графики сохраняются в папку images/ этого модуля.
"""

# %% [markdown]
# Шаг 0. Импорты и модель.

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))
from foundations import normal_power, type_m, mc_interval, mc_report

RNG = np.random.default_rng(20260909)
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

MU = 1240.0        # средняя выручка на юзера, руб. (нормальная учебная модель)
ALPHA = 0.05
ZZ = (stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(0.8)) ** 2  # (1.96+0.84)^2


# %% [markdown]
# Шаг 1. (а) Равные дисперсии, равные n — «мир, для которого придуман Стьюдент».
# (б) Неравные дисперсии + разные n: в тесте акция «двойной кэшбэк» —
# разброс чеков x3, и группа меньше. И обратная конфигурация.

# %%
REPS = 10_000
N1, SD1, N2, SD2 = 1000, 300.0, 100, 900.0

# (а) сигма одинаковые, n одинаковые
a = RNG.normal(MU, SD1, size=(REPS, N1))
b = RNG.normal(MU, SD1, size=(REPS, N1))
al_student_eq = np.mean(stats.ttest_ind(a, b, axis=1, equal_var=True).pvalue < ALPHA)
al_welch_eq = np.mean(stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue < ALPHA)

# (б) сигма и n разные: шумная группа маленькая
a = RNG.normal(MU, SD1, size=(REPS, N1))
b = RNG.normal(MU, SD2, size=(REPS, N2))
al_student_ne = np.mean(stats.ttest_ind(a, b, axis=1, equal_var=True).pvalue < ALPHA)
al_welch_ne = np.mean(stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue < ALPHA)

# (б-обратная) шумная группа большая
a_rev = RNG.normal(MU, SD2, size=(REPS, N1))
b_rev = RNG.normal(MU, SD1, size=(REPS, N2))
al_student_rev = np.mean(stats.ttest_ind(a_rev, b_rev, axis=1, equal_var=True).pvalue < ALPHA)
al_welch_rev = np.mean(stats.ttest_ind(a_rev, b_rev, axis=1, equal_var=False).pvalue < ALPHA)

# теория для (б): pooled-SE против честного SE -> фактическая alpha у Стьюдента
se_true = np.sqrt(SD1**2 / N1 + SD2**2 / N2)
sp2 = ((N1 - 1) * SD1**2 + (N2 - 1) * SD2**2) / (N1 + N2 - 2)
se_student = np.sqrt(sp2 * (1 / N1 + 1 / N2))
t_crit = stats.t.ppf(1 - ALPHA / 2, N1 + N2 - 2)
al_student_ne_th = 2 * stats.norm.sf(t_crit * se_student / se_true)

print("=" * 78)
print("1) СТЬЮДЕНТ (equal_var=True) vs УЭЛЧ: эмпирическая alpha под H0")
print("-" * 78)
print(f"   (а) сигма1=сигма2=300, n1=n2={N1}:")
print(f"       Стьюдент: {al_student_eq:.2%}   Уэлч: {al_welch_eq:.2%}   (номинал 5%)")
print(f"   (б) n1={N1} SD={SD1:.0f} (контроль), n2={N2} SD={SD2:.0f} (тест с акцией):")
print(f"       Стьюдент: {al_student_ne:.2%}   Уэлч: {al_welch_ne:.2%}")
print(f"       честный SE = {se_true:.2f}, а Стьюдент считает {se_student:.2f}")
print(f"       -> шум занижен в {se_true / se_student:.2f} раза, теория даёт alpha = {al_student_ne_th:.0%}")
print(f"   (б-обратная) n1={N1} SD={SD2:.0f}, n2={N2} SD={SD1:.0f}:")
print(f"       Стьюдент: {al_student_rev:.3%}   Уэлч: {al_welch_rev:.2%}")
print("       -> Стьюдент почти не отвергает: сломан «тихо», эффекты хоронятся")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
for ax, vals, title in (
    (axes[0], (al_student_eq, al_welch_eq), f"(а) равные дисперсии, n1=n2={N1}"),
    (axes[1], (al_student_ne, al_welch_ne), f"(б) SD {SD1:.0f} vs {SD2:.0f}, n {N1} vs {N2}"),
):
    ax.bar([0, 1], vals, color=["#C44E52", "#55A868"], width=0.55, alpha=0.9)
    ax.axhline(ALPHA, color="k", ls="--", lw=1.5)
    ax.text(1.35, ALPHA + 0.006, "номинал 5%", fontsize=9, ha="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Стьюдент\n(pooled)", "Уэлч"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.1%}", ha="center", fontweight="bold")
    ax.set_ylabel("фактическая доля p<0.05 под H0")
    ax.set_title(title, fontsize=10)
fig.suptitle("В этой нормальной модели Уэлч близок к номиналу; большие сигмы "
             "встречают маленькие n", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_2_2_student_welch.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_2_2_student_welch.png")

# %% [markdown]
# Шаг 2. (в) Парный vs непарный на ОДНИХ данных: NPS курьеров до/после
# обучения (n = 60, сдвиг +3, SD изменений 6, SD «до» 12).

# %%
N_PAIR = 60
before = RNG.normal(70, 12, N_PAIR)
change = RNG.normal(3, 6, N_PAIR)
after = before + change
diff = after - before

res_pair = stats.ttest_rel(after, before)
res_ind = stats.ttest_ind(after, before, equal_var=False)

corr = np.corrcoef(before, after)[0, 1]
sd_b, sd_a, sd_d = before.std(ddof=1), after.std(ddof=1), diff.std(ddof=1)
half_pair = stats.t.ppf(0.975, N_PAIR - 1) * sd_d / np.sqrt(N_PAIR)
se_ind = np.sqrt(sd_b**2 / N_PAIR + sd_a**2 / N_PAIR)
df_ind = se_ind**4 / ((sd_b**2 / N_PAIR) ** 2 / (N_PAIR - 1) + (sd_a**2 / N_PAIR) ** 2 / (N_PAIR - 1))
half_ind = stats.t.ppf(0.975, df_ind) * se_ind

n_pair_need = ZZ * sd_d**2 / diff.mean() ** 2
n_ind_need = ZZ * (sd_b**2 + sd_a**2) / diff.mean() ** 2  # на группу

print("=" * 78)
print("2) ПАРНЫЙ vs НЕПАРНЫЙ: NPS до/после обучения, n = %d" % N_PAIR)
print("-" * 78)
print(f"   средние до/после          : {before.mean():.1f} -> {after.mean():.1f} "
      f"(сдвиг {diff.mean():+.2f})")
print(f"   SD до / после / разностей : {sd_b:.1f} / {sd_a:.1f} / {sd_d:.1f}")
print(f"   корреляция до/после       : {corr:.2f}")
print(f"   парный t-тест   : t = {res_pair.statistic:+.2f}, p = {res_pair.pvalue:.2e}, "
      f"95% CI [{diff.mean() - half_pair:+.2f}; {diff.mean() + half_pair:+.2f}]")
print(f"   непарный t-тест : t = {res_ind.statistic:+.2f}, p = {res_ind.pvalue:.3f}, "
      f"95% CI [{diff.mean() - half_ind:+.2f}; {diff.mean() + half_ind:+.2f}]")
print(f"   n для 80% мощности: парному нужно ~{n_pair_need:.0f} юзеров,")
print(f"   непарному ~{n_ind_need:.0f} на группу — экономия ~x{n_ind_need / n_pair_need:.1f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
jx = np.column_stack([RNG.uniform(-0.18, 0.18, N_PAIR), RNG.uniform(0.82, 1.18, N_PAIR)])
for j, (b_, a_) in enumerate(zip(before, after)):
    axes[0].plot([jx[j, 0], jx[j, 1]], [b_, a_], color="gray", lw=0.6, alpha=0.5)
axes[0].scatter(jx[:, 0], before, s=34, color="#4C72B0", alpha=0.85, label="до")
axes[0].scatter(jx[:, 1], after, s=34, color="#DD8452", alpha=0.85, label="после")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels([f"до\nSD = {sd_b:.1f}", f"после\nSD = {sd_a:.1f}"])
axes[0].set_ylabel("NPS-скор")
axes[0].set_title("Непарный взгляд: два шумных облака\n(разброс юзеров глушит сдвиг)", fontsize=10)
axes[0].legend(fontsize=8)

axes[1].hist(diff, bins=14, color="#55A868", alpha=0.85)
axes[1].axvline(0, color="k", lw=1.2)
axes[1].axvline(diff.mean(), color="#C44E52", lw=2.4,
                label=f"средний сдвиг {diff.mean():+.2f}")
axes[1].set_xlabel(f"разность «после - до»\nSD разностей = {sd_d:.1f}")
axes[1].set_ylabel("число курьеров")
axes[1].set_title("Парный взгляд: одна тихая разность\n(юзер — сам себе контроль)", fontsize=10)
axes[1].legend(fontsize=8)

fig.suptitle("Парный дизайн = бесплатное снижение дисперсии (задел на CUPED, М3.5)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_2_2_paired.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_2_2_paired.png")

# %% [markdown]
# Шаг 3. (г) Скошенный чек: логнормал (медиана 700, сигма логарифмов 1.2,
# асимметрия ~11 — модель урока 1.3). Эмпирическая alpha t-теста при малых n.

# %%
MED, SL = 700.0, 1.2
TRUE_MEAN = MED * np.exp(SL**2 / 2)
SKEW = (np.exp(SL**2) + 2) * np.sqrt(np.exp(SL**2) - 1)
print("=" * 78)
print("3) СКОШЕННЫЙ ЧЕК (логнормал): эмпирическая alpha t-теста")
print("-" * 78)
print(f"   истинное среднее = {TRUE_MEAN:.1f} руб., асимметрия s = {SKEW:.1f}")
print(f"   эвристика Денга ~{100 * SKEW ** 2:,.0f}: ориентир, не гарантия/минимум")
print(f"   {'n':>6} | {'alpha одновыборочного':>21} | {'alpha двухвыборочного (A/A)':>26}")
grid = ((30, 40_000), (300, 20_000), (3000, 5_000))
rows = []
for n, reps in grid:
    x = RNG.lognormal(np.log(MED), SL, size=(reps, n))
    y = RNG.lognormal(np.log(MED), SL, size=(reps, n))
    a1 = np.mean(stats.ttest_1samp(x, TRUE_MEAN, axis=1).pvalue < ALPHA)
    a2 = np.mean(stats.ttest_ind(x, y, axis=1, equal_var=False).pvalue < ALPHA)
    rows.append((n, a1, a2))
    print(f"   {n:>6} | {a1:>21.2%} | {a2:>26.2%}")
    print("          one-sample:", mc_report(round(a1 * reps), reps))
    print("          two-sample:", mc_report(round(a2 * reps), reps))
print("   (номинал 5%; разность двух скошенных средних симметричнее —")
print("   двухвыборочный тест выносливее одновыборочного)")

# %%
fig, ax = plt.subplots(figsize=(8.2, 4.3))
xpos = np.arange(len(grid))
w = 0.36
mc_errors = []
for col in (1, 2):
    bounds = np.array([mc_interval(round(row[col] * reps), reps)
                       for row, (_, reps) in zip(rows, grid)])
    rates = np.array([row[col] for row in rows])
    mc_errors.append(np.array([rates - bounds[:, 0], bounds[:, 1] - rates]))
ax.bar(xpos - w / 2, [r[1] for r in rows], w, color="#C44E52", alpha=0.9,
       yerr=mc_errors[0], capsize=3,
       label="одновыборочный t (H0: среднее = истина)")
ax.bar(xpos + w / 2, [r[2] for r in rows], w, color="#4C72B0", alpha=0.9,
       yerr=mc_errors[1], capsize=3,
       label="двухвыборочный t Уэлча (A/A)")
ax.axhline(ALPHA, color="k", ls="--", lw=1.5)
ax.text(len(grid) - 0.55, ALPHA + 0.004, "номинал 5%", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels([f"n = {r[0]}" for r in rows])
ax.set_ylabel("фактическая доля p<0.05 под H0")
ax.set_title(f"Скошенный чек (асимметрия {SKEW:.0f}): проверяем уровень каждой процедуры\n"
             "Усы — 95% Monte Carlo CI; 100·s² не задаёт универсальный порог")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_2_2_skew_alpha.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_2_2_skew_alpha.png")

# %%
print()
print("=" * 78)
print("ГЛАВНОЕ (что должны увидеть):")
print("1) При равных n и дисперсиях в этой модели оба теста близки к номиналу.")
print("2) Неравные дисперсии + разные n: у Стьюдента alpha доезжает до ~37%")
print("   (в 37% A/A-тестов H0 ошибочно отвергнута) или падает почти до нуля —")
print("   в зависимости от того, где шум и где малая группа. Уэлч держит 5%:")
print("   equal_var=False снимает равенство дисперсий, но не все другие допущения.")
print("3) Парный анализ режет шум: SD разностей < SD групп при корреляции до/после.")
print(f"   Один и тот же сдвиг: парный p={res_pair.pvalue:.3f}, непарный p={res_ind.pvalue:.3f};")
print(f"   отношение n на группу к числу пар ≈{n_ind_need / n_pair_need:.1f}.")
print(f"4) При n=30: alpha одновыборочного={rows[0][1]:.2%}, двухвыборочного={rows[0][2]:.2%}.")
print("   Это результат конкретной модели, а не гарантия для других хвостов.")
print("   100·s² — ориентир; проверяем alpha, покрытие и мощность с MC-интервалами.")

for label, rate in [("Student equal", al_student_eq), ("Welch equal", al_welch_eq), ("Student unequal", al_student_ne), ("Welch unequal", al_welch_ne)]:
    print(label, mc_report(round(rate * REPS), REPS))
