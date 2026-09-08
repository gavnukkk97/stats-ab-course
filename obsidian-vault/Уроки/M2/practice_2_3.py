# -*- coding: utf-8 -*-
"""Практика 2.3 — Непараметрика и ранги (сквозной кейс «ЕдаДома»).

Не верь — проверь симуляцией.

Что делаем:
1) permutation-тест руками: пилот программы лояльности в малом городе
   (n = 20 на группу, выручка логнормальная с китами) — 5000 перемешиваний
   меток, p-value; сравнение с Уэлчем и Манном–Уитни + CI для вероятности
   превосходства P(B>A) бутстрапом;
2) киты + неравные группы: A/A-симуляция показывает, что у t-тестов
   «едет» уровень (фактическая доля p<0.05 != 5%), а Манн–Уитни
   и permutation держат номинал по построению;
3) мощность четырёх методов на логнормале: эффект в теле распределения
   (+25% всем) — ранги и лог-трансформация мощнее; эффект в хвосте
   (топ-10% «китов» тратят в 2.2 раза больше) — тесты среднего видят
   деньги, ранги слепы.

Запуск:  python3 practice_2_3.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# # Практика 2.3 — Непараметрика и ранги
# Правило курса: у каждого критерия своя оптика (среднее / порядок / любые
# статистики). Своими глазами убедимся, кто держит уровень на китах
# и кто кого по мощности — и когда.

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RNG = np.random.default_rng(23)  # seed -> результат воспроизводим
HERE = Path(__file__).resolve().parent
ALPHA = 0.05

# %% [markdown]
# ## Шаг 1. Permutation-тест руками: пилот в малом городе
# Пилот программы лояльности в городе N: 20 юзеров в контроле, 20 в пилоте.
# Метрика — выручка юзера за неделю: логнормал (медиана 800 руб., киты
# до ~40 000 руб.). Пилот мультипликативно поднимает траты всех участников
# в 2.2 раза. Идея permutation: если метка группы ни на что не влияет,
# то перемешивание ярлыков даёт «миры без эффекта» — и мы смотрим,
# насколько наш наблюдаемый эффект экстремален среди этих миров.

# %%
N_PILOT = 20
a = 800 * RNG.lognormal(0.0, 1.1, N_PILOT)          # контроль
b = 800 * 2.2 * RNG.lognormal(0.0, 1.1, N_PILOT)    # пилот: все траты x2.2

obs = b.mean() - a.mean()

B = 5_000
pool = np.concatenate([a, b])
# B случайных перестановок: сортируем случайную матрицу по строкам —
# каждая строка даёт случайный порядок элементов пула
perm_idx = np.argsort(RNG.random((B, len(pool))), axis=1)
perm_means = (
    pool[perm_idx][:, N_PILOT:].mean(axis=1)
    - pool[perm_idx][:, :N_PILOT].mean(axis=1)
)
# +1 в числителе и знаменателе: сам «наблюдаемый» мир тоже исход,
# p-value не может быть 0 на конечном числе перемешиваний
p_perm = (np.sum(np.abs(perm_means) >= np.abs(obs)) + 1) / (B + 1)

p_welch = stats.ttest_ind(a, b, equal_var=False).pvalue
mw = stats.mannwhitneyu(b, a, alternative="two-sided")
sup_hat = mw.statistic / (N_PILOT * N_PILOT)

# бутстрап-CI для вероятности превосходства P(B>A) = U/(mn)
B_BOOT = 3_000
ia = RNG.integers(0, N_PILOT, size=(B_BOOT, N_PILOT))
ib = RNG.integers(0, N_PILOT, size=(B_BOOT, N_PILOT))
sup_boot = (b[ib][:, :, None] > a[ia][:, None, :]).mean(axis=(1, 2))
ci_sup = np.quantile(sup_boot, [0.025, 0.975])

print("=" * 80)
print("1) Пилот в малом городе: n = 20 + 20, выручка логнормальная")
print(f"   контроль: mean = {a.mean():,.0f}, median = {np.median(a):,.0f}, max = {a.max():,.0f}")
print(f"   пилот:    mean = {b.mean():,.0f}, median = {np.median(b):,.0f}, max = {b.max():,.0f}")
print(f"   наблюдаемый эффект (разность средних) = {obs:+,.0f} руб.")
print(f"   permutation (5000 перемешиваний): p = {p_perm:.4f}")
print(f"   t Уэлча:                            p = {p_welch:.4f}")
print(f"   Манн–Уитни:                         p = {mw.pvalue:.4f}")
print(f"   вероятность превосходства P(B>A) = U/(mn) = {sup_hat:.2f},"
      f" бут-CI [{ci_sup[0]:.2f}; {ci_sup[1]:.2f}]")
print("   Читаем: permutation едва поймал денежный сдвиг; Уэлч не дотянул —")
print("   кит в пилоте разул SD; MW слабее всех: превосходство всего ~0.6-0.7,")
print("   рангам на n=20 нужно больше «разъезда» групп.")

# %%
fig, ax = plt.subplots(figsize=(8, 4.3))
ax.hist(perm_means, bins=60, color="#999", alpha=0.85,
        label="перемешанные метки (миры без эффекта)")
ax.axvline(obs, color="#C44E52", lw=3, label=f"наблюдаемый эффект = {obs:+,.0f} руб.")
ax.axvline(0, color="gray", ls=":", lw=1.5)
ax.set_title(f"Permutation-тест руками: p = {p_perm:.3f}\n"
             "доля миров без эффекта, где разность такая же экстремальная")
ax.set_xlabel("разность средних после перемешивания меток, руб.")
ax.set_ylabel("число миров")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "practice_2_3_permutation.png", dpi=150)
print("Сохранено: practice_2_3_permutation.png")

# %% [markdown]
# ## Шаг 2. Киты + неравные группы: у t-тестов едет уровень
# A/A-симуляция: эффектов нет, обе группы из одного логнормала
# (медиана 800 руб., sigma_log = 1.8 — жирные хвосты), но в контроль
# недолили трафика: 80 против 20. Условия t-теста (нормальность среднего
# через ЦПТ) при такой скошенности и малом n ещё не работают.
# Манн–Уитни и permutation обмениваемость меток держат по построению.

# %%
N1, N2 = 80, 20     # неравные группы — частая история (недолив трафика)
SIGMA_H0 = 1.8
WORLDS = 1_500
B_PERM = 300        # перемешиваний на мир (для скорости симуляции)


def perm_pvalue(x, y, B, rng):
    """p-value permutation-теста для разности средних."""
    pool = np.concatenate([x, y])
    obs_ = y.mean() - x.mean()
    idx = np.argsort(rng.random((B, len(pool))), axis=1)
    pm = pool[idx]
    perms = pm[:, len(x):].mean(axis=1) - pm[:, : len(x)].mean(axis=1)
    return (np.sum(np.abs(perms) >= np.abs(obs_)) + 1) / (B + 1)


cnt = {"t Стьюдента": 0, "t Уэлча": 0, "Манн–Уитни": 0, "permutation": 0}
for _ in range(WORLDS):
    a = 800 * RNG.lognormal(0.0, SIGMA_H0, N1)
    b = 800 * RNG.lognormal(0.0, SIGMA_H0, N2)  # A/A: эффектов нет
    cnt["t Стьюдента"] += stats.ttest_ind(a, b, equal_var=True).pvalue < ALPHA
    cnt["t Уэлча"] += stats.ttest_ind(a, b, equal_var=False).pvalue < ALPHA
    cnt["Манн–Уитни"] += (
        stats.mannwhitneyu(a, b, alternative="two-sided").pvalue < ALPHA
    )
    cnt["permutation"] += perm_pvalue(a, b, B_PERM, RNG) < ALPHA

alpha_hat = {k: v / WORLDS for k, v in cnt.items()}
print("=" * 80)
print(f"2) A/A с китами (lognormal, sigma={SIGMA_H0}), n1={N1} vs n2={N2}:"
      f" {WORLDS} миров, номинал alpha = {ALPHA:.0%}")
for k, v in alpha_hat.items():
    flag = "  <-- уровень поехал" if v > ALPHA + 0.015 else ""
    print(f"   {k:<14} фактическая доля p<0.05 = {v:.1%}{flag}")
print("   Манн–Уитни и permutation держат номинал: их нулевое распределение")
print("   строится из самих данных (ранги/перестановки), а не из формулы,")
print("   которая «привыкла» к нормальности.")

# %%
fig, ax = plt.subplots(figsize=(8, 4.3))
names = list(alpha_hat)
vals = [alpha_hat[n] for n in names]
colors = ["#C44E52", "#C44E52", "#55A868", "#55A868"]
ax.bar(names, vals, color=colors, alpha=0.9, width=0.55)
ax.axhline(ALPHA, color="black", ls="--", lw=1.6)
ax.text(3.35, ALPHA + 0.004, "номинал 5%", fontsize=9)
for i, v in enumerate(vals):
    ax.text(i, v + 0.006, f"{v:.1%}", ha="center", fontweight="bold")
ax.set_ylabel("фактическая доля p<0.05 под H0")
ax.set_ylim(0, max(vals) * 1.25)
ax.set_title("A/A с китами и неравными группами: t-тесты врёт про alpha,\n"
             "Манн–Уитни и permutation держат уровень")
fig.tight_layout()
fig.savefig(HERE / "practice_2_3_alpha.png", dpi=150)
print("Сохранено: practice_2_3_alpha.png")

# %% [markdown]
# ## Шаг 3. Мощность: эффект в теле против эффекта в хвосте
# Логнормальная выручка (медиана 800 руб., sigma_log = 1.0), n = 200 на группу.
# Два сценария эффекта:
# - «в теле»: все юзеры пилота тратят x1.25 — меняется и медиана, и среднее;
# - «в хвосте»: топ-10% «китов» пилота тратят x2.2 — среднее (деньги!)
#   растёт заметно, порядок почти не меняется.
# Методы: t Уэлча (raw), Манн–Уитни (ранги), permutation (среднее),
# t на логарифмах (log-трансформация — мостик к уроку 3.4).

# %%
N = 200
SIGMA_POW = 1.0
WORLDS_POW = 800
BODY_MULT = 1.25
TAIL_SHARE = 0.10
TAIL_MULT = 2.2
B_PERM_POW = 200

methods = ["t Уэлча (raw)", "Манн–Уитни", "permutation (mean)", "t на log"]
hits = {scen: {m: 0 for m in methods} for scen in ("тело (+25% всем)", "хвост (топ-10% x2.2)")}

for _ in range(WORLDS_POW):
    a = 800 * RNG.lognormal(0.0, SIGMA_POW, N)
    b_body = 800 * BODY_MULT * RNG.lognormal(0.0, SIGMA_POW, N)
    b_tail = 800 * RNG.lognormal(0.0, SIGMA_POW, N)
    whales = b_tail > np.quantile(b_tail, 1 - TAIL_SHARE)
    b_tail[whales] *= TAIL_MULT
    for scen, y in (("тело (+25% всем)", b_body), ("хвост (топ-10% x2.2)", b_tail)):
        hits[scen]["t Уэлча (raw)"] += stats.ttest_ind(a, y, equal_var=False).pvalue < ALPHA
        hits[scen]["Манн–Уитни"] += stats.mannwhitneyu(a, y, alternative="two-sided").pvalue < ALPHA
        hits[scen]["permutation (mean)"] += perm_pvalue(a, y, B_PERM_POW, RNG) < ALPHA
        hits[scen]["t на log"] += stats.ttest_ind(np.log(a), np.log(y), equal_var=False).pvalue < ALPHA

power = {scen: {m: v / WORLDS_POW for m, v in d.items()} for scen, d in hits.items()}
print("=" * 80)
print(f"3) Мощность на логнормале (n={N} на группу, {WORLDS_POW} миров)")
header = f"   {'метод':<22}{'эффект в теле':>15}{'эффект в хвосте':>17}"
print(header)
for m in methods:
    print(f"   {m:<22}{power['тело (+25% всем)'][m]:>15.1%}"
          f"{power['хвост (топ-10% x2.2)'][m]:>17.1%}")
print("   Эффект в теле: ранги и лог мощнее raw-среднего (хвосты душат t).")
print("   Эффект в хвосте: среднее видит деньги, ранги почти слепы —")
print("   кит и без того был выше всех, x2.2 не меняет его ранг заметно.")

# %%
fig, ax = plt.subplots(figsize=(8.6, 4.4))
xpos = np.arange(len(methods))
w = 0.38
body_vals = [power["тело (+25% всем)"][m] for m in methods]
tail_vals = [power["хвост (топ-10% x2.2)"][m] for m in methods]
ax.bar(xpos - w / 2, body_vals, w, color="#4C72B0", alpha=0.9, label="эффект в теле: +25% всем")
ax.bar(xpos + w / 2, tail_vals, w, color="#DD8452", alpha=0.9, label="эффект в хвосте: топ-10% x2.2")
ax.axhline(ALPHA, color="black", ls="--", lw=1.4)
ax.text(len(methods) - 0.55, ALPHA + 0.015, "alpha = 5%", fontsize=9)
for i, (vb, vt) in enumerate(zip(body_vals, tail_vals)):
    ax.text(i - w / 2, vb + 0.015, f"{vb:.0%}", ha="center", fontsize=9)
    ax.text(i + w / 2, vt + 0.015, f"{vt:.0%}", ha="center", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels(methods)
ax.set_ylabel("мощность")
ax.set_title("Кто что видит: ранги сильнее в теле распределения,\n"
             "тесты среднего — в хвосте (эффект-то в деньгах = в среднем)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "practice_2_3_power.png", dpi=150)
print("Сохранено: practice_2_3_power.png")

# %%
print()
print("=" * 80)
print("ГЛАВНОЕ (что должны увидеть):")
print(f"1) Permutation руками: эффект {obs:+,.0f} руб., p = {p_perm:.3f} — доля миров")
print("   без эффекта с такой же экстремальной разностью; (+1)/(B+1) не даёт")
print("   p=0 на конечном числе перемешиваний.")
print(f"2) На китах и малых n уровень t-тестов не гарантирован:")
print(f"   Стьюдент {alpha_hat['t Стьюдента']:.1%}, Уэлч {alpha_hat['t Уэлча']:.1%}"
      f" против номинала 5%; MW ({alpha_hat['Манн–Уитни']:.1%}) и permutation")
print(f"   ({alpha_hat['permutation']:.1%}) держат номинал по построению.")
print("3) Мощность зависит от того, ГДЕ эффект:")
print(f"   тело: MW {power['тело (+25% всем)']['Манн–Уитни']:.0%} и log {power['тело (+25% всем)']['t на log']:.0%}"
      f" против raw t {power['тело (+25% всем)']['t Уэлча (raw)']:.0%};")
print(f"   хвост: raw t {power['хвост (топ-10% x2.2)']['t Уэлча (raw)']:.0%} и permutation"
      f" {power['хвост (топ-10% x2.2)']['permutation (mean)']:.0%} видят деньги,")
print(f"   MW {power['хвост (топ-10% x2.2)']['Манн–Уитни']:.0%} слеп.")
print("4) У каждого теста своя гипотеза: среднее (деньги) / превосходство")
print("   (порядок) / любая статистика (permutation). Выбор теста = выбор")
print("   вопроса, а не «что значимее прокрасилось».")
