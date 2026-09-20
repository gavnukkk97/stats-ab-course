# -*- coding: utf-8 -*-
"""Практика 3.4 — Выбросы и трансформации метрик: винзоризация под микроскопом.

Сквозной кейс «ЕдаДома»: чек заказа ~ логнормальное распределение
(медиана 900 ₽, тяжёлый хвост). Не верь — проверь симуляцией.

Что делаем:
1) как киты убивают мощность: SD сырого чека против винзоризация 1/99 и 5/95
   и лог-трансформации; перевод в требуемую выборку (n ~ sigma^2);
2) эффект ЖИВЁТ В ХВОСТЕ (киты x1.5): raw-мощность мала, винзоризация 5/95
   прячет эффект ЦЕЛИКОМ («не отвергли H0» != «эффекта нет»), log тоже слеп;
   перцентильный (децильный) анализ показывает, где живёт эффект;
3) эффект ЖИВЁТ В ТЕЛЕ (+10% всем): винзоризация и лог видят его, raw почти нет;
4) раздельные трешолды по группам: в A/A ломают уровень (alpha ~ 18% вместо 5%),
   в A/B дают «число ни про что» — три политики, три разных ответа;
5) A/A-уровень всех трансформаций: правильные держат 5%, раздельный порог — нет.

Запуск из корня репозитория: python3 course/modules/M3/practice/practice_3_4.py
Графики сохраняются в папку images/ этого модуля.
"""

# %% [markdown]
# # Практика 3.4 — Винзоризация: мощность против эстиманда
# Главная мысль урока: трансформация метрики — это не «техническая чистка
# данных», а выбор того, ЧТО мы измеряем. Киты топ-5% дают +26% к средней
# выручке — а винзоризация 5/95 честно rapport «эффекта нет».

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(34)  # seed -> результат воспроизводим
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05

MU, SIGMA = np.log(900), 1.4          # медиана 900 ₽, хвост логнормали
N = 2_000                              # юзеров на группу
REPS = 1_500                           # миров для оценки мощности

# %% [markdown]
# ## Шаг 1. Киты и дисперсия: n пропорционально sigma^2
# Чек «ЕдаДома»: медиана 900 ₽, среднее ~1850 ₽ — хвост тащит. Считаем SD
# до и после трансформаций и переводим разницу в требуемую выборку:
# при фиксированном MDE нужно n ~ sigma^2 (урок 3.1).

# %%
POP = RNG.lognormal(MU, SIGMA, 2_000_000)      # «пре-период»: откуда берём пороги
LOW01 = np.percentile(POP, 1)
LOW05 = np.percentile(POP, 5)
CAP99 = np.percentile(POP, 99)
CAP95 = np.percentile(POP, 95)
sd_raw = POP.std()
sd_w99 = np.clip(POP, LOW01, CAP99).std()
sd_w95 = np.clip(POP, LOW05, CAP95).std()

df_sd = pd.DataFrame(
    {
        "SD чека, ₽": [sd_raw, sd_w99, sd_w95],
        "SD меньше, %": [0, 100 * (1 - sd_w99 / sd_raw), 100 * (1 - sd_w95 / sd_raw)],
        "n при том же численном Δ": [1.0, (sd_w99 / sd_raw) ** 2, (sd_w95 / sd_raw) ** 2],
    },
    index=["raw (сырой чек)", "винзоризация 1/99", "винзоризация 5/95"],
)
print("=" * 80)
print("1) Чек ~ LogNormal(медиана 900, sigma=1.4): среднее = {:,.0f} ₽, SD = {:,.0f} ₽".replace(",", " ").format(POP.mean(), sd_raw))
print(f"   пороги из пре-периода: P99 = {CAP99:,.0f} ₽, P95 = {CAP95:,.0f} ₽".replace(",", " "))
print(df_sd.round(2).to_string())
print(f"   log-метрика: SD(log чек) = {SIGMA:.2f} (безразмерная, монотонная трансформация)")
print("   Винзоризация 5/95 режет SD в ~2.5 раза -> при том же численном Δ на новой шкале нужно ~6 раз меньше юзеров; эффект фичи тоже меняется.")
print("   (X5: винзоризация снижает нужную выборку до 59%; снос датчиков — до 10 раз, симулятор урок 6.)")

# %%
fig, ax = plt.subplots(figsize=(8.6, 4.2))
bins = np.logspace(0, 5.2, 140)
ax.hist(POP[:100_000], bins=bins, color="#4C72B0", alpha=0.85)
ax.axvline(CAP95, color="#DD8452", lw=2.4, ls="--", label=f"P95 = {CAP95:,.0f} ₽ (5/95)".replace(",", " "))
ax.axvline(CAP99, color="#C44E52", lw=2.4, label=f"P99 = {CAP99:,.0f} ₽ (1/99)".replace(",", " "))
ax.set_xscale("log")
ax.set_xlabel("чек, ₽ (log-шкала)")
ax.set_ylabel("юзеров")
ax.set_title(f"Хвост тащит SD: {sd_raw:,.0f} ₽; винзоризация 5/95 -> {sd_w95:,.0f} ₽ (n меньше в {(sd_raw/sd_w95)**2:.1f} раза)".replace(",", " "))
ax.legend()
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_3_4_sd.png", dpi=150)
print("Сохранено: practice_3_4_sd.png")

# %% [markdown]
# ## Шаг 2. Эффект живёт В ХВОСТЕ: киты x1.5, а винзоризация говорит «эффекта нет»
# Фича «ЕдаДома» (корпоративные клиенты): топ-5% юзеров (по базовому уровню,
# как если бы выделяли китов по пре-периоду) стали заказывать в 1.5 раза
# больше. Истинный эффект ~ +26% выручки. Смотрим, что видит каждая метрика:
# raw / винзоризация 1/99 / винзоризация 5/95 / log. Пороги — ЕДИНЫЕ на обе
# группы, посчитанные по объединённой выборке (как и положено).

# %%
WHALE_MULT = 1.5


def world_tail(n=N, with_effect=True, return_baseline=False):
    """Один мир: контроль и тест; киты (топ-5% базового уровня) x1.5."""
    c = RNG.lognormal(MU, SIGMA, n)
    t0 = RNG.lognormal(MU, SIGMA, n)
    t = t0 * np.where(t0 > CAP95, WHALE_MULT, 1.0) if with_effect else t0
    return (c, t, t0) if return_baseline else (c, t)


def variants(c, t):
    """Фиксированные двусторонние пороги из независимого пре-периода."""
    hi99, hi95 = CAP99, CAP95
    return {
        "raw": (t, c),
        "win 1/99": (np.clip(t, LOW01, hi99), np.clip(c, LOW01, hi99)),
        "win 5/95": (np.clip(t, LOW05, hi95), np.clip(c, LOW05, hi95)),
        "log": (np.log(t), np.log(c)),
    }


p_cnt = {k: 0 for k in variants(*world_tail())}
eff_sum = {k: 0.0 for k in p_cnt}
true_eff = 0.0
for _ in range(REPS):
    c, t = world_tail()
    true_eff += t.mean() - c.mean()
    for k, (tv, cv) in variants(c, t).items():
        p_cnt[k] += stats.ttest_ind(tv, cv, equal_var=False).pvalue < ALPHA
        eff_sum[k] += tv.mean() - cv.mean()
true_eff /= REPS

print("=" * 80)
print(f"2) Эффект В ХВОСТЕ: топ-5% китов x{WHALE_MULT}, n = {N:,} на группу, {REPS:,} миров".replace(",", " "))
print(f"   истинный эффект = {true_eff:,.0f} ₽ ({true_eff / POP.mean() * 100:.0f}% средней выручки!)".replace(",", " "))
for k in p_cnt:
    kept = eff_sum[k] / REPS / true_eff * 100
    print(f"   {k:9s}: мощность = {p_cnt[k] / REPS:5.0%}   сохранили эффект: {kept:5.0f}%")
print("   ВИНОРИЗАЦИЯ 5/95 СПРЯТАЛА ЭФФЕКТ ПОЛНОСТЬЮ: мощность = уровень A/A, оценка ~ 0.")
print("   «не отвергли H0» != «эффекта нет» — мы просто перестали его измерять.")
print("   log тоже слеп: эффект у 5% юзеров почти не сдвигает E[log чек].")

# %% [markdown]
# ### Перцентильный анализ: где живёт эффект
# Среднее по децилям (юзеры ранжированы по базовому уровню чека): если разница
# сосредоточена в верхних децилях — винзоризация противопоказана, эффект
# «живёт» именно там, куда она режет.

# %%
edges = np.percentile(POP, np.linspace(0, 100, 11))
W = 300
dec_diff = np.zeros(10)
for _ in range(W):
    c, t, t0 = world_tail(return_baseline=True)
    for d in range(10):
        m = (c > edges[d]) & (c <= edges[d + 1])
        dec_diff[d] += t[(t0 > edges[d]) & (t0 <= edges[d + 1])].mean() - c[m].mean()
dec_diff /= W

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
axes[0].bar(range(1, 5), [p_cnt[k] / REPS * 100 for k in p_cnt],
            color=["#C44E52", "#DD8452", "#C44E52", "#55A868"], width=0.55)
axes[0].axhline(5, color="black", ls="--", lw=1.5)
axes[0].set_xticks(range(1, 5))
axes[0].set_xticklabels(list(p_cnt), fontsize=8)
axes[0].set_ylabel("мощность, % миров с p<0.05")
axes[0].set_title("Эффект в хвосте (+26% выручки): что видит каждая метрика")
axes[1].bar(range(1, 11), dec_diff, color=["#4C72B0"] * 8 + ["#DD8452", "#C44E52"], width=0.62)
axes[1].axhline(0, color="black", lw=1.2)
axes[1].set_xticks(range(1, 11))
axes[1].set_xticklabels([f"D{d+1}" for d in range(10)], fontsize=9)
axes[1].set_xlabel("дециль базового чека")
axes[1].set_ylabel("средняя разница тест-контроль, ₽")
axes[1].set_title("Перцентильный анализ: эффект живёт в D9-D10 — винзоризация режет именно там")
fig.suptitle("Эффект на китах: raw слаб, 5/95 прячет его целиком, децили показывают где он", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_3_4_tail.png", dpi=150)
print("Сохранено: practice_3_4_tail.png")
print(f"   децильные разницы, ₽: " + ", ".join(f"D{d+1}={dec_diff[d]:+.0f}" for d in range(10)))

# %% [markdown]
# ## Шаг 3. Эффект живёт В ТЕЛЕ: +10% всем — трансформации работают
# Другая фича (скидка на доставку): все юзеры заказывают на ~10% больше.
# Здесь винзоризация и лог снижают шум, НЕ потеряв эффект (мультипликативный
# эффект равномерно размазан) — выигрыш в мощности настоящий.

# %%
BODY_MULT = 1.10
p_cnt_b = {k: 0 for k in ["raw", "win 1/99", "win 5/95", "log"]}
eff_b = {k: 0.0 for k in p_cnt_b}
for _ in range(REPS):
    c = RNG.lognormal(MU, SIGMA, N)
    t = RNG.lognormal(MU, SIGMA, N) * BODY_MULT
    for k, (tv, cv) in variants(c, t).items():
        p_cnt_b[k] += stats.ttest_ind(tv, cv, equal_var=False).pvalue < ALPHA
        eff_b[k] += tv.mean() - cv.mean()

print("=" * 80)
print(f"3) Эффект В ТЕЛЕ: всем x{BODY_MULT}, n = {N:,} на группу, {REPS:,} миров".replace(",", " "))
print(f"   истинный (raw) эффект ~ +{(BODY_MULT - 1) * POP.mean():.0f} ₽ (~+{(BODY_MULT - 1) * 100:.0f}% выручки)")
for k in p_cnt_b:
    avg = eff_b[k] / REPS
    shown = f"{avg:+.3f} (в log-единицах)" if k == "log" else f"{avg:+,.0f} ₽".replace(",", " ")
    print(f"   {k:9s}: мощность = {p_cnt_b[k] / REPS:5.0%}   средняя оценка: {shown}")
print("   Винзоризация и log видят эффект заметно лучше raw: SD меньше, эффект размазан по всем.")
print("   Но заметьте: абсолютная оценка эффекта в ₽ у трансформированных метрик ДРУГАЯ (эстиманд сместился).")

# %% [markdown]
# ## Шаг 4. Раздельные трешолды по группам = сломанный тест
# Считать порог отдельно в каждой группе по данным эксперимента — грубая
# ошибка: порог становится функцией данных, которые уже содержат эффект
# (и шум экстремальных квантилей). Демонстрация в A/A (уровень alpha)
# и в A/B с эффектом на китах (три политики — три разных ответа).

# %%
REPS_AA = 3_000
levels = {"raw": 0, "win верхний P95 (общий, пре-период)": 0, "win верхний P95 (общий, pooled)": 0,
          "win верхний P95 (РАЗДЕЛЬНЫЕ)": 0, "log": 0, "ранги (Манн-Уитни)": 0}
for _ in range(REPS_AA):
    a, b = RNG.lognormal(MU, SIGMA, N), RNG.lognormal(MU, SIGMA, N)   # A/A: эффекта нет
    pooled = np.percentile(np.concatenate([a, b]), 95)
    levels["raw"] += stats.ttest_ind(a, b, equal_var=False).pvalue < ALPHA
    levels["win верхний P95 (общий, пре-период)"] += stats.ttest_ind(
        np.clip(a, None, CAP95), np.clip(b, None, CAP95), equal_var=False).pvalue < ALPHA
    levels["win верхний P95 (общий, pooled)"] += stats.ttest_ind(
        np.clip(a, None, pooled), np.clip(b, None, pooled), equal_var=False).pvalue < ALPHA
    levels["win верхний P95 (РАЗДЕЛЬНЫЕ)"] += stats.ttest_ind(
        np.clip(a, None, np.percentile(a, 95)), np.clip(b, None, np.percentile(b, 95)),
        equal_var=False).pvalue < ALPHA
    levels["log"] += stats.ttest_ind(np.log(a), np.log(b), equal_var=False).pvalue < ALPHA
    levels["ранги (Манн-Уитни)"] += stats.mannwhitneyu(a, b).pvalue < ALPHA

print("=" * 80)
print(f"4a) A/A-уровень ({REPS_AA:,} миров, номинал 5%):".replace(",", " "))
for k, v in levels.items():
    flag = "  <-- ЛОМАЕТ УРОВЕНЬ" if v / REPS_AA > 0.075 else ""
    print(f"   {k:32s}: {v / REPS_AA:5.1%}{flag}")

# A/B на китах: три политики порогов -> три разных ответа
REPS_AB = 2_000
est = {"raw": [], "win верхний P95 общий (пре-период)": [], "win верхний P95 раздельные": []}
for _ in range(REPS_AB):
    c, t = world_tail()
    est["raw"].append(t.mean() - c.mean())
    est["win верхний P95 общий (пре-период)"].append(
        np.clip(t, None, CAP95).mean() - np.clip(c, None, CAP95).mean())
    est["win верхний P95 раздельные"].append(
        np.clip(t, None, np.percentile(t, 95)).mean() - np.clip(c, None, np.percentile(c, 95)).mean())

print(f"\n4b) A/B, эффект на китах (истина по raw ~ {true_eff:,.0f} ₽), {REPS_AB:,} миров:".replace(",", " "))
tab = pd.DataFrame(
    {k: [np.mean(v), np.std(v)] for k, v in est.items()},
    index=["E[оценка эффекта], ₽", "SD оценки, ₽"],
).T.round(0)
print(tab.to_string())
print("   Три политики — три разных «эффекта»: у раздельных порогов оценка дрейфует")
print("   вместе с шумом квантилей; такой оценщик требует отдельного estimand и SE с учётом оценки порога.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
ks = list(levels)
cols = ["#C44E52" if levels[k] / REPS_AA > 0.075 else "#55A868" for k in ks]
axes[0].barh(range(len(ks)), [levels[k] / REPS_AA * 100 for k in ks], color=cols, alpha=0.9)
axes[0].axvline(5, color="black", ls="--", lw=1.5)
axes[0].set_yticks(range(len(ks)))
axes[0].set_yticklabels(ks, fontsize=8)
axes[0].set_xlabel("доля p<0.05 в A/A, %")
axes[0].set_title("A/A-уровень: раздельные трешолды ломают alpha,\nобщий порог держит 5%")
for i, k in enumerate(est):
    axes[1].hist(est[k], bins=55, alpha=0.65, label=f"{k}: E={np.mean(est[k]):+.0f} ₽")
axes[1].axvline(0, color="black", lw=1.2)
axes[1].axvline(true_eff, color="black", ls="--", lw=1.5, label=f"истина (raw) = {true_eff:+.0f} ₽")
axes[1].set_xlabel("оценка эффекта, ₽")
axes[1].set_title("A/B на китах: политики порогов дают разные ответы")
axes[1].legend(fontsize=8)
fig.suptitle("Один порог на обе группы (и из пре-периода) — иначе тест ломается", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_3_4_thresholds.png", dpi=150)
print("Сохранено: practice_3_4_thresholds.png")

# %%
print()
print("=" * 80)
print("ГЛАВНОЕ (что должны увидеть):")
print(f"1) SD чека {sd_raw:,.0f} ₽ -> винзоризация 5/95 {sd_w95:,.0f} ₽: при том же численном Δ на новой шкале n меньше в {(sd_raw / sd_w95) ** 2:.1f} раза".replace(",", " "))
print(f"2) Эффект на китах (+{true_eff / POP.mean() * 100:.0f}% выручки): raw-мощность {p_cnt['raw'] / REPS:.0%},")
print(f"   5/95 прячет эффект ЦЕЛИКОМ (мощность {p_cnt['win 5/95'] / REPS:.0%} = A/A-уровень, сохранено ~0%), log тоже слеп.")
print("   Перцентильный анализ (децили) показывает, где живёт эффект — делать до вывода «эффекта нет».")
print(f"3) Эффект в теле (+10% всем): мощность raw {p_cnt_b['raw'] / REPS:.0%} -> "
      f"5/95 {p_cnt_b['win 5/95'] / REPS:.0%}, log {p_cnt_b['log'] / REPS:.0%} — трансформации работают,")
print("   но абсолютная оценка эффекта меняется: вы решили измерять ДРУГУЮ величину.")
print(f"4) Раздельные трешолды: в A/A alpha = {levels['win верхний P95 (РАЗДЕЛЬНЫЕ)'] / REPS_AA:.0%} (номинал 5%);")
print("   в A/B оценка зависит от шума квантилей — обычный t-test игнорирует оценку порога. Для простого рецепта порог один на обе группы,")
print("   посчитан по независимой истории и зафиксирован ДО запуска; pooled-порог проверяем отдельно.")
print("5) A/A-уровень — обязательная проверка любой трансформации: в этом генераторе raw, общий win, log, ранги дали уровень около 5%.")
