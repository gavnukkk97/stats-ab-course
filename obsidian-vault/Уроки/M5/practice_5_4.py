# -*- coding: utf-8 -*-
"""Практика 5.4 — Бандиты: Thompson sampling вместо теста
(сквозной кейс «ЕдаДома»).

Не верь — проверяй симуляцией.

Что делаем:
1) ТРИ РУКИ (креативы баннера: конверсия 10% / 12% / 14%), горизонт 30 000
   показов, 500 прогонов. Четыре политики на одних и тех же руках:
   - fixed A/B (равный сплит 1/3 на весь тест, победитель — в конце);
   - epsilon-greedy (eps=0.1: 90% — лучший наблюдаемый, 10% — случайный);
   - UCB1 («оптимизм в лицо неопределённости»: mean + sqrt(2 ln t / n));
   - Thompson sampling (сэмплируем тету из бета-апостериоров, показываем
     руку с максимальным сэмплом).
   Метрики: кумулятивный regret (цена обучения) и доля показов лучшей руки;
2) ЭПИЗОД «ПРОМО-КАМПАНИЯ»: 10 000 показов — сколько конверсий теряет
   каждая политика против оракула (лучший креатив всегда);
3) НЕСТАЦИОНАРНОСТЬ: лучший креатив «приедается» на середине (14% -> 6%).
   Наивный Thompson (вся история в апостериоре) против скользящего окна
   (апостериор по последним W показам): скорость реакции и regret.

Запуск:  python3 practice_5_4.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# # Практика 5.4 — Thompson sampling vs epsilon-greedy vs fixed A/B
# Регret = сумма (лучшее возможное − показанное). Это «цена обучения»:
# сколько конверсий мы недополучили, пока выясняли, какой креатив лучше.

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RNG = np.random.default_rng(54)  # seed -> результат воспроизводим
HERE = Path(__file__).resolve().parent

ARMS = np.array([0.10, 0.12, 0.14])   # конверсия трёх креативов
K = len(ARMS)
BEST = ARMS.max()
HORIZON = 30_000
RUNS = 500
EVERY = 50                            # частота чекпоинтов


# %%
def run_policy(rng, name, horizon=HORIZON, runs=RUNS, arms=ARMS, window=None):
    """Прогон одной политики. Возвращает regret и долю лучшей руки
    на чекпоинтах (средние по прогонам), плюс историю выбора/награды."""
    k = len(arms)
    best = arms.max()
    succ = np.zeros((runs, k))          # успехи по рукам (или в окне)
    fail = np.zeros((runs, k))
    seen = np.zeros((runs, k))          # сколько раз показали руку
    pulls_best = np.zeros(runs)         # показов лучшей руки
    reg = np.zeros(runs)                # кумулятивный regret
    cp_reg, cp_share, cp_t = [], [], []
    # история нужна только для скользящего окна
    hist_c = np.zeros((horizon, runs), dtype=np.int8) if window else None
    hist_r = np.zeros((horizon, runs), dtype=np.int8) if window else None

    for t in range(horizon):
        if name == "fixed":                       # равный сплит по расписанию
            arm = np.full(runs, t % k)
        elif name == "eps":                       # epsilon-greedy
            if t < k:                             # сперва каждую руку по разу
                arm = np.full(runs, t % k)
            else:
                means = succ / seen
                arm = np.where(rng.random(runs) < 0.10,
                               rng.integers(0, k, runs),
                               means.argmax(axis=1))
        elif name == "ucb":
            if t < k:
                arm = np.full(runs, t % k)
            else:
                bonus = np.sqrt(2.0 * np.log(t + 1) / seen)
                arm = (succ / seen + bonus).argmax(axis=1)
        elif name == "ts":                        # Thompson sampling
            samples = rng.beta(succ + 1.0, fail + 1.0)
            arm = samples.argmax(axis=1)
        else:                                     # ts + скользящее окно
            samples = rng.beta(succ + 1.0, fail + 1.0)
            arm = samples.argmax(axis=1)

        p = arms[arm]
        rew = (rng.random(runs) < p)
        idx = np.arange(runs)
        succ[idx, arm] += rew
        fail[idx, arm] += ~rew
        seen[idx, arm] += 1
        reg += best - p
        pulls_best += (arm == arms.argmax())
        if window:
            hist_c[t] = arm
            hist_r[t] = rew
            if t >= window:                       # выбывает из окна
                old_c = hist_c[t - window]
                old_r = hist_r[t - window]
                succ[idx, old_c] -= old_r
                fail[idx, old_c] -= ~old_r
        if (t + 1) % EVERY == 0:
            cp_reg.append(reg.mean())
            cp_share.append(pulls_best.mean() / (t + 1))
            cp_t.append(t + 1)
    return dict(name=name, t=np.array(cp_t), regret=np.array(cp_reg),
                share=np.array(cp_share), final_regret=reg.mean(),
                final_pulls_best=pulls_best.mean())


# %% [markdown]
# ## Шаг 1. Regret: сколько конверсий стоило «учиться»

# %%
res = [run_policy(RNG, name) for name in ("fixed", "eps", "ucb", "ts")]
names_ru = dict(fixed="fixed A/B (сплит 1/3)", eps="epsilon-greedy (10%)",
                ucb="UCB1", ts="Thompson sampling")

print("=" * 88)
print(f"1) Три руки {list(ARMS)}, горизонт {HORIZON:,} показов, {RUNS} прогонов")
for r in res:
    i10 = np.searchsorted(r["t"], 10_000)
    print(f"   {names_ru[r['name']]:<26} regret@10k = {r['regret'][i10]:>7.1f}"
          f"   regret@30k = {r['regret'][-1]:>7.1f}   "
          f"доля лучшей руки к концу = {r['share'][-1]:.0%}")
print("   fixed A/B: треть трафика всегда у ХУДШЕЙ руки и треть у средней —")
print("   regret растёт ЛИНЕЙНО и никогда не останавливается.")
print("   бандиты: regret растёт логарифмически — заплатили за обучение и едим")
print("   почти только лучшую руку.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
for r, color in zip(res, ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]):
    ax.plot(r["t"], r["regret"], lw=2.2, color=color, label=names_ru[r["name"]])
ax.set_xlabel("показ, №")
ax.set_ylabel("кумулятивный regret (потерянные конверсии)")
ax.set_title("Цена обучения: у fixed A/B regret растёт линейно,\n"
             "у бандитов — логарифмически (Thompson ближе всех к нулю)")
ax.legend(fontsize=9)

ax = axes[1]
for r, color in zip(res, ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]):
    ax.plot(r["t"], r["share"] * 100, lw=2.2, color=color, label=names_ru[r["name"]])
ax.axhline(100 / K, color="grey", ls=":", lw=1.2, label="случайный выбор = 33%")
ax.set_xlabel("показ, №")
ax.set_ylabel("доля показов лучшей руки, %")
ax.set_title("Переливание трафика: Thompson быстро сводит показы к лучшему\nкреативу, fixed A/B до конца теста раздаёт поровну")
ax.set_ylim(0, 102)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(HERE / "practice_5_4_regret.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Шаг 2. Эпизод «промо-кампания»: 10 000 показов, реклама живёт часы
# Оракул (сразу лучший креатив): 0.14 * 10 000 = 1 400 конверсий.
# Считаем потерянные конверсии каждой политики = regret на 10-м тысячном
# показе + для fixed A/B ещё риск выбрать не того победителя в конце.

# %%
N_PROMO = 10_000
oracle = BEST * N_PROMO
i10 = np.searchsorted(res[0]["t"], N_PROMO)
promo = pd.DataFrame([
    dict(политика=names_ru[r["name"]],
         конверсии=f"{oracle - r['regret'][i10]:.0f}",
         потеряно=f"-{r['regret'][i10]:.0f}")
    for r in res
])
# риск «не того победителя» у fixed: лучший по счётчикам после равного сплита
n_arm = N_PROMO // K
conv = np.stack([RNG.binomial(n_arm, p, 4000) for p in ARMS], axis=1)
wrong_winner = (conv.argmax(axis=1) != ARMS.argmax()).mean()

print("=" * 88)
print(f"2) Промо-кампания: {N_PROMO:,} показов, оракул = {oracle:.0f} конверсий")
print(promo.to_string(index=False))
print(f"   fixed A/B ещё и выберет не того победителя в {wrong_winner:.1%} кампаний"
      " (нужен второй тест,\n   чтобы это исправить); бандиту «победитель» не нужен —"
      " он уже раздавал трафик правильно.")

# %%
fig, ax = plt.subplots(figsize=(8.6, 4.4))
labels = [names_ru[r["name"]] for r in res]
lost = [r["regret"][i10] for r in res]
bars = ax.barh(labels[::-1], lost[::-1],
               color=["#C44E52", "#55A868", "#DD8452", "#4C72B0"])
for bar, v in zip(bars, lost[::-1]):
    ax.annotate(f"−{v:.0f} конверсий", (v, bar.get_y() + bar.get_height() / 2),
                va="center", ha="left", fontsize=10)
ax.axvline(oracle * 0 + 0, color="k", lw=0.8)
ax.set_xlabel("потерянные конверсии за 10 000 показов (регрет против оракула)")
ax.set_title("Пока тестировали — теряли: fixed A/B отдаёт оракулу каждую пятую\n"
             "«недополученную» конверсию, Thompson — единицы")
ax.set_xlim(0, max(lost) * 1.25)
fig.tight_layout()
fig.savefig(HERE / "practice_5_4_promo.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Шаг 3. Нестационарность: лучший креатив «приедается» на середине
# Рука 0: 14% -> 6% с показа 10 000 (аудитория устала от промо).
# Рука 1: стабильные 10%. Наивный Thompson помнит ВСЮ историю — его
# апостериор по руке 0 «тугой» (тысячи показов) и сдвигается медленно.
# Thompson со скользящим окном W=1500 забывает старьё и перестраивается.

# %%
HOR = 20_000
SWITCH = 10_000
ARMS_NS = np.array([0.14, 0.10])       # рука 0 портится в середине


def run_ns(rng, window):
    arms = ARMS_NS.copy()
    succ = np.zeros((RUNS, 2))
    fail = np.zeros((RUNS, 2))
    reg = np.zeros(RUNS)
    pull0 = np.zeros(RUNS)
    cp_t, cp_share, cp_reg = [], [], []
    hist_c = np.zeros((HOR, RUNS), dtype=np.int8)
    hist_r = np.zeros((HOR, RUNS), dtype=np.int8)
    for t in range(HOR):
        p_now = arms.copy()
        if t >= SWITCH:
            p_now[0] = 0.06
        samples = rng.beta(succ + 1.0, fail + 1.0)
        arm = samples.argmax(axis=1)
        p = p_now[arm]
        rew = rng.random(RUNS) < p
        idx = np.arange(RUNS)
        succ[idx, arm] += rew
        fail[idx, arm] += ~rew
        reg += p_now.max() - p
        pull0 += arm == 0
        hist_c[t] = arm
        hist_r[t] = rew
        if window and t >= window:
            old_c = hist_c[t - window]
            old_r = hist_r[t - window]
            succ[idx, old_c] -= old_r
            fail[idx, old_c] -= ~old_r
        if (t + 1) % EVERY == 0:
            cp_t.append(t + 1)
            cp_share.append(pull0.mean() / (t + 1))
            cp_reg.append(reg.mean())
    return dict(t=np.array(cp_t), share=np.array(cp_share),
                regret=np.array(cp_reg), final=reg.mean())


ns_naive = run_ns(RNG, window=None)
ns_win = run_ns(RNG, window=1500)

i_switch = np.searchsorted(ns_naive["t"], SWITCH)
second_half = slice(i_switch, None)
r2_naive = ns_naive["regret"][-1] - ns_naive["regret"][i_switch]
r2_win = ns_win["regret"][-1] - ns_win["regret"][i_switch]
# доля руки 0 в свежих показах после слома (скользящее среднее по 2000)
sh0 = ns_naive["share"].copy()
sh0_after = []
for res_ns in (ns_naive, ns_win):
    # пересчёт доли руки 0 на последних показах: производная от кумулятивной
    t = res_ns["t"]
    cum = res_ns["share"] * t
    fresh = (cum[-1] - cum[np.searchsorted(t, HOR - 2000)]) / 2000
    sh0_after.append(fresh)

print("=" * 88)
print("3) Нестационарность: рука 0 (14%) портится до 6% на показе "
      f"{SWITCH:,}; рука 1 = 10%")
print(f"   regret за ВТОРУЮ половину (после слома):")
print(f"     наивный Thompson (вся история):        {r2_naive:6.0f} конверсий")
print(f"     Thompson с окном W=1500:               {r2_win:6.0f} конверсий")
print(f"   доля руки 0 на последних 2000 показов: наивный {sh0_after[0]:.0%}"
      f" против окна {sh0_after[1]:.0%}")
print("   наивный бандит «помнит славное прошлое»: тысячи показов делают")
print("   апостериор тугим, и он показывает испорченный креатив ещё долго;")
print("   скользящее окно жертвует стабильностью ради скорости реакции")
print("   (компромисс: чем меньше окно — тем быстрее реакция и тем выше")
print("   шум на стационарных участках).")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
for res_ns, tag, color in [(ns_naive, "наивный Thompson", "#4C72B0"),
                           (ns_win, "Thompson, окно W=1500", "#C44E52")]:
    ax.plot(res_ns["t"], res_ns["regret"], lw=2.2, color=color, label=tag)
ax.axvline(SWITCH, color="grey", ls="--", lw=1.4, label="рука 0 испортилась (14%→6%)")
ax.set_xlabel("показ, №")
ax.set_ylabel("кумулятивный regret")
ax.set_title("Слом стационарности: окно перестраивается за ~W показов,\nнаивный бандит ещё долго верит «тугому» апостериору")
ax.legend(fontsize=9)

ax = axes[1]
for res_ns, tag, color in [(ns_naive, "наивный Thompson", "#4C72B0"),
                           (ns_win, "Thompson, окно W=1500", "#C44E52")]:
    t = res_ns["t"]
    cum = res_ns["share"] * t
    w = 40                               # 40 чекпоинтов = 2000 показов
    fresh = (cum[w:] - cum[:-w]) / (t[w:] - t[:-w])
    ax.plot(t[w:], fresh * 100, lw=2.0, color=color, label=tag)
ax.axvline(SWITCH, color="grey", ls="--", lw=1.4, label="рука 0 испортилась")
ax.set_xlabel("показ, №")
ax.set_ylabel("доля показов руки 0, % (окно 1000)")
ax.set_title("Реакция на слом: окно уводит трафик к руке 1 (~10%),\nнаивный Thompson продолжает лить на испорченную руку")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(HERE / "practice_5_4_nonstationary.png", dpi=150)
plt.close(fig)

print("\nГрафики: practice_5_4_regret.png, practice_5_4_promo.png, "
      "practice_5_4_nonstationary.png")
