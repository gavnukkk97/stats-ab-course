# -*- coding: utf-8 -*-
"""Практика 6.2 — Potential outcomes: мир, который мы не увидели.

Принцип курса: не верь — проверь симуляцией. В симуляции нам доступна
«божественная перспектива» — мы генерируем ОБА потенциальных исхода
Y(0) и Y(1) для каждого юзера, а «наблюдаем» только один по назначению.
Четыре блока (кейс «ЕдаДома Прайм» — программа лояльности):

1) фундаментальная проблема: таблица двух миров; наивная разность средних
   при самоселекции участников != ATE (декомпозиция: наив = ATE + смещение
   отбора + смещение самого факта участия... см. вывод);
2) рандомизация: то же DGP, но вступление назначается монеткой —
   разность средних восстанавливает ATE; распределение оценки по 2000 миров;
3) SUTVA-нарушение: спилловор внутри пар друзей (купон «приведи друга»);
   рандомизация честная, а поюзерный сплит смещает оценку общего эффекта
   (связь с уроком 4.3: доля утечки гамма);
4) ATT vs ATE: эффект неоднороден (кому частая доставка — тем и пользы
   больше) + самоселекция -> наивная оценка отвечает не на тот вопрос.

Запуск:  python3 practice_6_2.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# # Практика 6.2 — Два мира: Y(0) и Y(1) в симуляции
# В реальных данных второй мир всегда потерян. В симуляции — нет: генерируем
# оба, «наблюдаем» один. Это единственный способ увидеть смещение своими глазами.

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(62)  # seed -> результат воспроизводим
HERE = Path(__file__).resolve().parent

# %% [markdown]
# ## Шаг 1. Фундаментальная проблема: два мира, наблюдаем один
#
# DGP: U ~ N(0,1) — «аппетит» юзера (платёжеспособность + любовь к доставке).
# Программу «Прайм» люди выбирают сами: p(вступить) = sigmoid(1.5*U) —
# голодные и платёжеспособные вступают чаще (self-selection).
# Исход — заказы в месяц ПОСЛЕ решения. Истинный эффект +1,5 заказа всем.

# %%
N = 5_000
U = RNG.normal(0, 1, N)
TRUE_ATE = 1.5

Y0 = 8 + 3 * U + RNG.normal(0, 2, N)   # мир, где юзер НЕ в «Прайме»
Y1 = Y0 + TRUE_ATE + 0 * U              # мир, где юзер в «Прайме» (эффект одинаков)

# «Наблюдаем» по самоселекции
T_selfish = (RNG.random(N) < 1 / (1 + np.exp(-1.5 * U))).astype(int)
Y_obs = np.where(T_selfish == 1, Y1, Y0)

po = pd.DataFrame({
    "U": U.round(2), "T": T_selfish,
    "Y0_невидимый": Y0.round(1), "Y1_невидимый": Y1.round(1),
    "Y_наблюдаемый": Y_obs.round(1),
    "ITE=Y1-Y0": (Y1 - Y0).round(2),
})
print("=" * 78)
print("1) ДВА МИРА (первые 6 юзеров; в реальности серые столбцы потеряны)")
print("-" * 78)
print(po.head(6).to_string(index=False))

ate_true = np.mean(Y1 - Y0)
naive = Y_obs[T_selfish == 1].mean() - Y_obs[T_selfish == 0].mean()
sel_bias_Y0 = Y0[T_selfish == 1].mean() - Y0[T_selfish == 0].mean()

print("-" * 78)
print(f"   истинный ATE (знаем только мы)            : {ate_true:+.2f} заказа/мес")
print(f"   наивная разность E[Y|T=1] - E[Y|T=0]      : {naive:+.2f} заказа/мес")
print(f"   смещение отбора E[Y0|T=1] - E[Y0|T=0]     : {sel_bias_Y0:+.2f}")
print(f"   проверка: наив = ATT({np.mean((Y1 - Y0)[T_selfish == 1]):+.2f}) "
      f"+ смещение({sel_bias_Y0:+.2f}) = {np.mean((Y1 - Y0)[T_selfish == 1]) + sel_bias_Y0:+.2f}")
print("   -> наивная разность завышена в разы: в «Прайм» и без него шли самые")
print("      голодные. Корреляция ~ 4 заказа, причинность ~ 1,5.")

# %% [markdown]
# ## Шаг 2. Рандомизация: та же DGP, но вступление назначает монетка
#
# Ключ: T больше не зависит ни от U, ни от (Y0, Y1) — группы уравнены
# «в среднем по всем характеристикам, включая ненаблюдаемые».
# Прогоняем 2000 параллельных миров: где сидит оценка разности средних?

# %%
REPS = 2000
ests_rand, ests_selfish = np.empty(REPS), np.empty(REPS)
for r in range(REPS):
    T_rand = RNG.integers(0, 2, N)  # честная монетка 50/50
    Yr = np.where(T_rand == 1, Y1, Y0)
    ests_rand[r] = Yr[T_rand == 1].mean() - Yr[T_rand == 0].mean()
    Ts = (RNG.random(N) < 1 / (1 + np.exp(-1.5 * U))).astype(int)
    Ys = np.where(Ts == 1, Y1, Y0)
    ests_selfish[r] = Ys[Ts == 1].mean() - Ys[Ts == 0].mean()

cover = np.mean(
    [stats.t.interval(0.95, 2500 - 2,
                      loc=e, scale=np.sqrt(np.var(RNG.choice(Y1, 0) if False else [0]) + 1))[0]
     for e in []]  # placeholder, считаем покрытие ниже честнее
) if False else None

# честное покрытие: CI для разности средних в каждом мире
cover_cnt = 0
for r in range(300):  # 300 миров достаточно для оценки покрытия
    T_rand = RNG.integers(0, 2, N)
    Yr = np.where(T_rand == 1, Y1, Y0)
    a, b = Yr[T_rand == 1], Yr[T_rand == 0]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    est = a.mean() - b.mean()
    lo, hi = est - 1.96 * se, est + 1.96 * se
    cover_cnt += (lo <= TRUE_ATE <= hi)

print("=" * 78)
print("2) РАНДОМИЗАЦИЯ: 2000 миров, одна и та же DGP")
print("-" * 78)
print(f"   самоселекция : средняя оценка {ests_selfish.mean():+.2f} "
      f"(разброс sd {ests_selfish.std():.2f}) — стабильно ЗАВЫШЕНА")
print(f"   рандомизация : средняя оценка {ests_rand.mean():+.2f} "
      f"(sd {ests_rand.std():.2f}) — центр на истинном ATE {TRUE_ATE:+.1f}")
print(f"   95% CI накрывает истинный ATE в {cover_cnt / 300:.0%} миров из 300")
print("   -> рандомизация не убирает ШУМ (sd остался), она убирает СМЕЩЕНИЕ.")
print("      Именно за это А/Б-тесты — золотой стандарт (мост в М4).")

# %%
fig, ax = plt.subplots(figsize=(10.6, 4.4))
bins = np.linspace(0.5, 4.8, 90)
ax.hist(ests_selfish, bins=bins, color="#C44E52", alpha=0.75,
        label=f"самоселекция: среднее {ests_selfish.mean():+.2f} — смещение "
              f"{ests_selfish.mean() - TRUE_ATE:+.2f}")
ax.hist(ests_rand, bins=bins, color="#4C72B0", alpha=0.75,
        label=f"рандомизация: среднее {ests_rand.mean():+.2f}")
ax.axvline(TRUE_ATE, color="k", lw=2.5, ls="--", label=f"истинный ATE = {TRUE_ATE:+.1f}")
ax.set_xlabel("оценка разности средних E[Y|T=1] − E[Y|T=0], заказов/мес")
ax.set_ylabel("число миров из 2000")
ax.set_title("Одна и та же DGP: самоселекция стабильно завышает, рандомизация бьёт в цель")
ax.legend(fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "practice_6_2_randomization.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_6_2_randomization.png")

# %% [markdown]
# ## Шаг 3. SUTVA-нарушение: спилловор в парах друзей
#
# Купон «приведи друга»: тестовый юзер может поделиться промокодом с другом
# из КОНТРОЛЯ — чужое лечение течёт в твой исход. Y_i зависит не только от
# T_i, но и от T_друга -> no interference нарушено. Рандомизация честная,
# но поюзерный сплит отвечает не на вопрос «что будет, если дать ВСЕМ».

# %%
N_PAIR = 4_000
DELTA = 2.0   # прямой эффект купона на самого юзера
BETA = 1.2    # утечка к другу, если друг в тесте
U3 = RNG.normal(0, 1, N_PAIR)
BASE = 8 + 3 * U3

# поюзерная рандомизация 50/50 внутри пар
T1 = RNG.integers(0, 2, N_PAIR)
T2 = RNG.integers(0, 2, N_PAIR)
eps = RNG.normal(0, 1.5, 2 * N_PAIR)

# наблюдаемый исход: своё лечение + лечение друга (интерференция!)
Y1_obs = BASE + DELTA * T1 + BETA * T2 + eps[:N_PAIR]
Y2_obs = BASE + DELTA * T2 + BETA * T1 + eps[N_PAIR:]
T_all = np.concatenate([T1, T2])
Y_all = np.concatenate([Y1_obs, Y2_obs])

est_userlevel = Y_all[T_all == 1].mean() - Y_all[T_all == 0].mean()
direct_effect = DELTA
total_effect = DELTA + BETA  # что реально изменится при раскатке на всех
leak = BETA / (DELTA + BETA)

# оценка по кластерам-парам: сравниваем пары, где оба T, с парами, где никто
both = (T1 == 1) & (T2 == 1)
none_ = (T1 == 0) & (T2 == 0)
est_pairlevel = ((Y1_obs + Y2_obs)[both].mean() - (Y1_obs + Y2_obs)[none_].mean()) / 2

print("=" * 78)
print("3) SUTVA: спилловор купона «приведи друга» (рандомизация честная!)")
print("-" * 78)
print(f"   прямой эффект купона (своё лечение)      : {direct_effect:+.2f}")
print(f"   утечка к другу (чужое лечение, бета)      : {BETA:+.2f}")
print(f"   истинный ОБЩИЙ эффект при раскатке на всех: {total_effect:+.2f}")
print(f"   поюзерный сплит видит                    : {est_userlevel:+.2f} "
      f"= {direct_delta_formula if False else ''}{DELTA} + {BETA}*P(друг в тесте=0.5)")
print(f"   -> занижение на {total_effect - est_userlevel:.2f} заказа "
      f"({leak:.0%} эффекта утекло; в уроке 4.3 это доля гамма)")
print(f"   лечение по парам (обе версии vs ни одной): {est_pairlevel:+.2f} — кластер")
print("      случай рандомизации лечит: пары-кластеры не делят друга с контролем.")
print("   Механика: наивная оценка = дельта + бета*E[T_друга] = "
      f"{DELTA} + {BETA}*0.5 = {DELTA + 0.5 * BETA:.2f} — что и наблюдаем.")

# %%
fig, ax = plt.subplots(figsize=(10.2, 4.2))
labels = ["прямой эффект\n(свой купон)", "видит поюзерный сплит\n(свой + половина чужого)",
          "истинный общий эффект\n(полная раскатка)", "видит кластерный тест\n(пары целиком)"]
vals = [direct_effect, est_userlevel, total_effect, est_pairlevel]
colors = ["#C9C9C9", "#DD8452", "#4C72B0", "#55A868"]
bars = ax.bar(labels, vals, color=colors, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:+.2f}",
            ha="center", fontsize=11, fontweight="bold")
ax.axhline(total_effect, color="#4C72B0", ls=":", lw=1.5)
ax.set_ylabel("оценённый эффект, заказов/мес")
ax.set_ylim(0, 3.8)
ax.set_title(f"Интерференция: честный поюзерный А/Б занижает эффект на "
             f"{leak:.0%} (утечка гамма из урока 4.3)")
ax.tick_params(axis="x", labelsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "practice_6_2_sutva.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_6_2_sutva.png")

# %% [markdown]
# ## Шаг 4. ATT vs ATE: эффект неоднороден + самоселекция
#
# Теперь эффект у каждого свой: tau_i = -1 + 2.2*U — частым гостям «Прайм»
# даёт много, редким — в минус (доставка по подписке не окупается).
# Вступают снова голодные -> наивная оценка отвечает «сколько пользы ТЕМ,
# КТО ПРИШЁЛ», а продакт спрашивает «что будет, если продавать ВСЕМ».

# %%
N4 = 20_000
U4 = RNG.normal(0, 1, N4)
TAU = -1 + 2.2 * U4                 # индивидуальный эффект
Y0b = 8 + 3 * U4 + RNG.normal(0, 2, N4)
Y1b = Y0b + TAU                     # Y(1) = Y(0) + tau_i
T4 = (RNG.random(N4) < 1 / (1 + np.exp(-1.5 * U4))).astype(int)
Y4 = np.where(T4 == 1, Y1b, Y0b)

ate = np.mean(TAU)
att = np.mean(TAU[T4 == 1])
naive4 = Y4[T4 == 1].mean() - Y4[T4 == 0].mean()

print("=" * 78)
print("4) ATT vs ATE: неоднородный эффект + самоселекция в «Прайм»")
print("-" * 78)
print(f"   истинный ATE  (эффект для случайного юзера)      : {ate:+.2f} заказа/мес")
print(f"   истинный ATT  (эффект для вступивших)            : {att:+.2f} заказа/мес")
print(f"   наивная разность по наблюдаемым                  : {naive4:+.2f} заказа/мес")
print(f"   доля вступивших: {T4.mean():.0%}; их средний аппетит U = {U4[T4 == 1].mean():+.2f} "
      f"против {U4[T4 == 0].mean():+.2f} у остальных")
print("   -> программа убыточна в среднем (ATE < 0), но выгодна пришедшим (ATT > 0):")
print("      «навязать всем» и «дать тем, кто сам пришёл» — РАЗНЫЕ решения.")
print("      Наивная оценка не равна даже ATT: смещение по Y(0) сверху "
      f"(+{(Y0b[T4 == 1].mean() - Y0b[T4 == 0].mean()):+.2f}).")

# %%
fig, ax = plt.subplots(figsize=(10.8, 4.4))
grid = np.linspace(-3.2, 3.2, 400)
ax.plot(grid, -1 + 2.2 * grid, color="k", lw=2, label="индивидуальный эффект τ(U) = −1 + 2,2·U")
ax.scatter(U4[T4 == 1][::40], TAU[T4 == 1][::40], s=5, alpha=0.25,
           color="#C44E52", label="вступили в «Прайм» (голодающие, τ выше)")
ax.scatter(U4[T4 == 0][::80], TAU[T4 == 0][::80], s=5, alpha=0.25,
           color="#4C72B0", label="не вступили")
ax.axhline(ate, color="#4C72B0", lw=2, ls="--", label=f"ATE = {ate:+.2f}")
ax.axhline(att, color="#C44E52", lw=2, ls="--", label=f"ATT = {att:+.2f}")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("аппетит юзера U")
ax.set_ylabel("эффект «Прайма», заказов/мес")
ax.set_title("Эффект неоднороден, а вступают те, кому выгодно: ATE и ATT — разные ответы")
ax.legend(fontsize=9, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "practice_6_2_att_ate.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_6_2_att_ate.png")
