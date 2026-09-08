# -*- coding: utf-8 -*-
"""Практика 2.4 — Множественные сравнения (сквозной кейс «ЕдаДома»).

Не верь — проверь симуляцией.

Что делаем:
1) 1000 симуляций A/A-теста «ЕдаДома» с 20 метриками: как часто хотя бы
   одна метрика «прокрашивается» (FWER) без поправок и с поправками
   Бонферрони / Холма / BH;
2) Холм и BH руками (скорректированные p-value) + сверка со
   statsmodels.stats.multitest.multipletests;
3) кривая FWER от числа гипотез m: теория 1-(1-alpha)^m против симуляции;
   плюс коррелированные метрики (метрики дашборда зависимы!) — формула
   завышает, но проблема никуда не девается.

Запуск:  python3 practice_2_4.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# # Практика 2.4 — Множественные сравнения
# «20 метрик — одна прокрасится» — проверим буквально: 1000 параллельных
# вселенных, в каждой A/A-тест (эффектов нет!) на 20 метриках.

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

RNG = np.random.default_rng(42)  # seed -> результат воспроизводим
HERE = Path(__file__).resolve().parent
ALPHA = 0.05
M = 20          # метрик в тесте
WORLDS = 1_000  # параллельных вселенных

# %% [markdown]
# ## Шаг 1. A/A с 20 метриками: сколько миров «откроют» эффект
# Каждая метрика в каждом мире — честный z-тест при H0: p-value равномерно
# на [0, 1]. Сами p-value рисуем через z ~ N(0,1): p = 2·(1 - Ф(|z|)).

# %%
z = RNG.standard_normal((WORLDS, M))
p = 2 * stats.norm.sf(np.abs(z))  # под H0: p ~ U(0,1)

hits_raw = (p < ALPHA).sum(axis=1)          # сколько метрик прокрасилось в мире
fwer_raw = np.mean(hits_raw >= 1)           # доля миров хотя бы с одной прокраской


def holm_adjust(pvals):
    """Холм (step-down): скорректированные p-value, контролирует FWER."""
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    order = np.argsort(pvals)
    adj_sorted = (m - np.arange(m)) * pvals[order]   # порог i-го: p(i) <= alpha/(m-i+1)
    adj_sorted = np.maximum.accumulate(adj_sorted)   # шаг вниз: p_adj монотонны по рангу
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


def bh_adjust(pvals):
    """Бенджамини–Хохберг: скорректированные p-value, контролирует FDR."""
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    order = np.argsort(pvals)
    i = np.arange(1, m + 1)
    adj_sorted = m * pvals[order] / i                 # порог i-го: p(i) <= (i/m)*q
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


fwer = {
    "без поправки": fwer_raw,
    "Бонферрони": np.mean([np.any(pw * M <= ALPHA) for pw in p]),
    "Холм": np.mean([np.any(holm_adjust(pw) <= ALPHA) for pw in p]),
    "BH (FDR)": np.mean([np.any(bh_adjust(pw) <= ALPHA) for pw in p]),
}
mean_hits = hits_raw.mean()
max_hits = hits_raw.max()

print("=" * 80)
print(f"1) A/A-тест «ЕдаДома»: {M} метрик, {WORLDS} миров, эффектов НЕТ")
for k, v in fwer.items():
    print(f"   {k:<14} доля миров с >=1 значимой метрикой = {v:.1%}")
print(f"   среднее число прокрасок на мир = {mean_hits:.2f} (~alpha*m = {ALPHA * M:.1f})")
print(f"   максимум прокрасок в одном мире = {max_hits}")
print("   Без поправки «открытие» почти гарантировано (теория: 1-0.95^20 = "
      f"{1 - 0.95 ** M:.0%}).")
print("   С любой поправкой FWER/FDR возвращается к номиналу 5%.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
ax.hist(hits_raw, bins=np.arange(-0.5, M + 1.5), color="#4C72B0", alpha=0.9)
ax.axvline(hits_raw.mean(), color="#C44E52", lw=2.4,
           label=f"в среднем {hits_raw.mean():.1f} прокраски")
ax.set_xlabel(f"число метрик с p<{ALPHA} в одном A/A-мире")
ax.set_ylabel("число миров")
ax.set_title("Все прокраски — ложные: эффектов нет")
ax.legend(fontsize=9)
ax = axes[1]
names = list(fwer)
vals = [fwer[n] for n in names]
ax.bar(names, vals, color=["#C44E52", "#55A868", "#55A868", "#55A868"], alpha=0.9)
ax.axhline(ALPHA, color="black", ls="--", lw=1.6)
ax.text(3.05, ALPHA + 0.02, "номинал 5%", fontsize=9)
for i, v in enumerate(vals):
    ax.text(i, v + 0.015, f"{v:.0%}", ha="center", fontweight="bold")
ax.set_ylabel("доля миров с >=1 значимой метрикой")
ax.set_ylim(0, 1.0)
ax.set_title(f"FWER: {fwer['без поправки']:.0%} без поправки -> номинал с поправкой")
fig.suptitle("A/A x 20 метрик: без поправки «открываем» эффект в 2 из 3 миров",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE / "practice_2_4_aa20.png", dpi=150)
print("Сохранено: practice_2_4_aa20.png")

# %% [markdown]
# ## Шаг 2. Холм и BH руками + сверка со statsmodels
# Скорректированное p-value — это «наименьший alpha, при котором гипотеза
# ещё отвергается данной процедурой». Сравниваем с эталоном.

# %%
demo_5 = np.array([0.60, 0.04, 0.20, 0.01, 0.02])  # мини-таблица из урока
rng2 = np.random.default_rng(77)
demo_20 = np.concatenate([rng2.uniform(0, 0.02, 6),    # 6 метрик с реальным эффектом
                          rng2.uniform(0.1, 0.9, 14)]) # 14 шумовых

print("=" * 80)
print("2) Холм и BH руками против statsmodels")
for name, pv in (("5 p-value из урока", demo_5), ("20 метрик (6 с эффектом)", demo_20)):
    my_holm, my_bh = holm_adjust(pv), bh_adjust(pv)
    for method, mine in (("holm", my_holm), ("fdr_bh", my_bh)):
        _, ref, _, _ = multipletests(pv, alpha=ALPHA, method=method)
        diff = np.max(np.abs(mine - ref))
        status = "OK" if diff < 1e-12 else "РАСХОЖДЕНИЕ!"
        print(f"   {name}: {method:<6} max|моё - statsmodels| = {diff:.1e}  [{status}]")
    print(f"   отвергаются: Бонферрони {int(np.sum(pv * pv.size <= ALPHA))},"
          f" Холм {int(np.sum(my_holm <= ALPHA))},"
          f" BH {int(np.sum(my_bh <= ALPHA))} из {pv.size}")

# %% [markdown]
# ## Шаг 3. Кривая FWER от m: теория, симуляция и коррелированные метрики
# Формула 1-(1-alpha)^m предполагает НЕЗАВИСИМЫЕ тесты. Метрики дашборда
# зависимы (одни и те же юзеры): z_j = sqrt(rho)*общий_фактор + sqrt(1-rho)*шум.
# Корреляция снижает FWER, но не отменяет проблему.

# %%
m_grid = np.array([1, 2, 3, 5, 10, 20, 30, 50])
WORLDS3 = 3_000
RHO = 0.5
fwer_theory = 1 - (1 - ALPHA) ** m_grid
fwer_ind, fwer_corr = [], []
for m in m_grid:
    z_ind = RNG.standard_normal((WORLDS3, m))
    latent = RNG.standard_normal((WORLDS3, 1))
    z_corr = np.sqrt(RHO) * latent + np.sqrt(1 - RHO) * RNG.standard_normal((WORLDS3, m))
    fwer_ind.append(np.mean(np.any(2 * stats.norm.sf(np.abs(z_ind)) < ALPHA, axis=1)))
    fwer_corr.append(np.mean(np.any(2 * stats.norm.sf(np.abs(z_corr)) < ALPHA, axis=1)))

print("=" * 80)
print(f"3) FWER от m (alpha={ALPHA:.0%}, {WORLDS3} миров; корреляция rho={RHO})")
print(f"   {'m':>4}{'теория 1-(1-a)^m':>18}{'симуляция (независ.)':>21}{'симуляция (rho=0.5)':>21}")
for m, ft, fi, fc in zip(m_grid, fwer_theory, fwer_ind, fwer_corr):
    print(f"   {m:>4}{ft:>18.1%}{fi:>21.1%}{fc:>21.1%}")
print("   Коррелированные метрики красятся «пачками» -> FWER ниже независимой")
print("   формулы, но при m=20 и rho=0.5 всё равно сильно выше 5%.")

# %%
fig, ax = plt.subplots(figsize=(8.4, 4.4))
mm = np.arange(1, 51)
ax.plot(mm, 1 - (1 - ALPHA) ** mm, lw=2.4, color="#C44E52",
        label="теория (независимые): 1-(1-0.05)^m")
ax.plot(m_grid, fwer_ind, "o", ms=7, color="#4C72B0", label="симуляция: независимые")
ax.plot(m_grid, fwer_corr, "s", ms=7, color="#55A868", label=f"симуляция: коррелированные (rho={RHO})")
ax.axhline(ALPHA, color="black", ls="--", lw=1.4)
ax.annotate("m=20: 64%", xy=(20, 1 - 0.95**20), xytext=(26, 0.45),
            arrowprops=dict(arrowstyle="->", color="#C44E52"),
            color="#C44E52", fontweight="bold")
ax.set_xlabel("число гипотез m")
ax.set_ylabel("FWER — вероятность >=1 ложного открытия")
ax.set_title("Чем больше проверок, тем вернее «открытие»:\nкорреляция метрик замедляет рост, но не останавливает")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(HERE / "practice_2_4_fwer.png", dpi=150)
print("Сохранено: practice_2_4_fwer.png")

# %%
print()
print("=" * 80)
print("ГЛАВНОЕ (что должны увидеть):")
print(f"1) A/A x {M} метрик без поправок: {fwer['без поправки']:.0%} миров с «открытием»")
print(f"   (теория 64%); в среднем {mean_hits:.1f} ложная прокраска на мир.")
print("2) Бонферрони/Холм возвращают FWER к 5%; BH контролирует не FWER,")
print("   а FDR — при полном H0 это тоже ~5%, разница проявится на мощности.")
print("3) Ручные holm_adjust/bh_adjust совпали со statsmodels до 1e-12.")
print(f"4) FWER растёт с m: m=50 -> {1 - 0.95 ** 50:.0%} (теория); корреляция")
print("   метрик снижает кривую, но не спасает.")
print("5) Лекарство проще формул: фиксируй метрики и момент проверки")
print("   в дизайн-доке до запуска (а подглядывание = множественность во времени,")
print("   урок 3.7).")
