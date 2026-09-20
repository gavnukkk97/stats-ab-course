# -*- coding: utf-8 -*-
"""Практика 3.7 — Подглядывание и последовательное тестирование
(сквозной кейс «ЕдаДома»).

Не верь — проверяй симуляцией.

Что делаем:
1) A/A-тест с ЕЖЕДНЕВНЫМ подглядыванием: эмпирическая альфа против
   номинальных 5%; кривая инфляции от числа проверок (1..28);
   политики «остановиться при p<0,05», «продлить после неудачного взгляда»,
   «прокрас + подтверждение финалом»;
2) упрощённый mSPRT (смешанный SPRT на нормальных данных, смесь по сдвигу):
   always-valid p-value = min(1, min(1/Lambda)); проверяем, что при ежедневных
   проверках A/A-уровень держится (~5%), а граница решимости зависит от времени;
3) сколько дней до детекции фиксированного эффекта: fixed-horizon
   (план по мощности) vs sequential (ежедневный mSPRT) — скорость
   против цены гибкости.

Запуск из корня репозитория: python3 course/modules/M3/practice/practice_3_7.py
Графики сохраняются в папку images/ этого модуля.
"""

# %% [markdown]
# # Практика 3.7 — peeking, mSPRT и цена гибкости
# Трафик «ЕдаДома»: 200 юзеров в день на руку, метрика в единицах сигмы
# (sigma = 1). Смотрим тест каждый день. p-value — двухсторонний z-тест
# по накопленным данным (сигма известна; t ничего не меняет по сути).

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(37)  # seed -> результат воспроизводим
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05

N_DAY = 200        # юзеров в день на руку
DAYS = 60          # максимум дней
WORLDS = 20_000    # число симуляционных миров
K_DAYS = np.arange(1, DAYS + 1)
V_K = 2.0 / (K_DAYS * N_DAY)     # Var(дельта-шляпки) к дню k (sigma=1)


def simulate_paths(rng, effect=0.0, worlds=WORLDS, days=DAYS):
    """Ежедневные приращения разности средних -> накопленные z и p."""
    daily = rng.normal(effect, np.sqrt(2.0 / N_DAY), size=(worlds, days))
    k_days = np.arange(1, days + 1)
    v_days = 2.0 / (k_days * N_DAY)
    x = np.cumsum(daily, axis=1) / k_days            # дельта-шляпка к дню k
    z = x / np.sqrt(v_days)
    p = 2 * stats.norm.sf(np.abs(z))
    return x, z, p


# %% [markdown]
# ## Шаг 1. A/A + ежедневное подглядывание: насколько взлетает альфа
# Правило «смотрю каждый день; как только p<0,05 — останавливаюсь и качу».
# Это НЕ один тест, а максимум по дням: альфа = P(min_k p_k < 0,05).

# %%
x0, z0, p0 = simulate_paths(RNG, effect=0.0)
hit = p0 < ALPHA

alpha_curve = hit.cumsum(axis=1).astype(bool).mean(axis=0)   # P(прокрас <= K дней)

a_fixed28 = hit[:, 27].mean()
a_daily28 = hit[:, :28].any(axis=1).mean()
a_daily60 = hit.any(axis=1).mean()

# «продление после взгляда»: день 14 посмотрели; прокрас — стоп-победа;
# не прокрас — продлеваем тест вдвое (до 42 дней) и решаем по финалу
peek14_win = hit[:, 13]
extend_win = (~hit[:, 13]) & hit[:, 41]
a_extend = (peek14_win | extend_win).mean()

# «прокрас на день 14 — что скажет финал 28-го дня?»
mask14 = hit[:, 13]
confirm_share = hit[mask14, 27].mean()
exagg14 = (np.abs(x0[mask14, 13]).mean() / np.abs(x0[mask14, 27]).mean()
           if mask14.sum() else np.nan)

print("=" * 88)
print("1) A/A-тест (эффекта нет), 200 юзеров/день на руку, {} миров".format(WORLDS))
print(f"   посмотреть ОДИН раз (день 28):            альфа = {a_fixed28:.1%}  (номинал 5%)")
print(f"   смотреть каждый день, стоп при p<0,05:    альфа = {a_daily28:.1%}  (28 проверок)")
print(f"   то же до 60 дней:                         альфа = {a_daily60:.1%}  (60 проверок)")
print(f"   взгляд на 14-й день + продление до 42:    альфа = {a_extend:.1%}  (2 проверки, но вторая")
print("   'заряжена' первой) — продление НЕ возвращает номинал")
print(f"   среди миров с прокрасом на 14-й день: значимы и на 28-й только {confirm_share:.0%};")
print(f"   оценка эффекта в день прокраса в среднем в {exagg14:.1f} раза дальше от нуля, чем финальная —")
print("   ранние прокрасы это преувеличенные 'эффекты' (type M на минималках).")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(K_DAYS[:28], alpha_curve[:28] * 100, lw=2.6, color="#C44E52",
             label="эмпирическая альфа при K ежедневных проверках")
axes[0].axhline(5, color="#55A868", ls="--", lw=1.8, label="номинал 5%")
axes[0].axvline(28, color="gray", ls=":", lw=1.2)
axes[0].annotate(f"28 проверок: {a_daily28:.0%}", xy=(28, a_daily28 * 100),
                 xytext=(14, a_daily28 * 100 + 4), color="#C44E52", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#C44E52"))
axes[0].set_xlabel("число ежедневных проверок")
axes[0].set_ylabel("доля ложных «эффектов» в A/A, %")
axes[0].set_title("Инфляция альфа от подглядывания")
axes[0].legend(fontsize=9, loc="lower right")

pol = ["один взгляд\n(день 28)", "каждый день\nстоп при p<0,05\n(28 дней)",
       "каждый день\n(60 дней)", "день 14 +\nпродление до 42"]
vals = [a_fixed28, a_daily28, a_daily60, a_extend]
axes[1].bar(range(4), np.array(vals) * 100, color=["#55A868", "#C44E52", "#C44E52", "#DD8452"],
            width=0.6, alpha=0.9)
for i, v in enumerate(vals):
    axes[1].text(i, v * 100 + 0.7, f"{v:.1%}", ha="center", fontweight="bold")
axes[1].axhline(5, color="black", ls="--", lw=1.6)
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(pol, fontsize=8.5)
axes[1].set_ylabel("эмпирическая альфа, %")
axes[1].set_title("Политики взгляда на один и тот же A/A-тест")
fig.suptitle("«Смотреть каждый день и остановиться при p<0,05» — не тест, а генератор ложных побед",
             fontsize=11.5, fontweight="bold")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_3_7_peeking.png", dpi=150)
print("Сохранено: practice_3_7_peeking.png")

# %% [markdown]
# ## Шаг 2. Упрощённый mSPRT: always-valid p-value
# Смешанное правдоподобие: сдвиг delta ~ N(0, tau^2), сигма известна.
#   Lambda_k = sqrt(v_k/(tau^2+v_k)) * exp( x_k^2 * tau^2 / (2 v_k (tau^2+v_k)) )
# Lambda — неотрицательный мартингейл под H0, поэтому по неравенству Вилля
#   P( max_k Lambda_k >= 1/alpha ) <= alpha   при ЛЮБОМ правиле остановки.
# Always-valid p-value: p_tilde_k = min(1, min_{j<=k} 1/Lambda_j) — его можно
# смотреть хоть каждый день. Плата — граница решимости растёт со временем.

# %%
TAU = 0.10  # масштаб смеси ~ ожидаемому эффекту (в долях сигмы)


def msprt_loglr(x, v, tau=TAU):
    """log смешанного LR: p_mix(x)/p_0(x), p_mix = N(0, v + tau^2), p_0 = N(0, v)."""
    return 0.5 * np.log(v / (v + tau**2)) + x**2 * tau**2 / (2 * v * (v + tau**2))


loglr = msprt_loglr(x0, V_K)
loglr_cummax = np.maximum.accumulate(loglr, axis=1)
p_av = np.exp(-loglr_cummax)          # always-valid p-value (min 1/Lambda)
p_av = np.clip(p_av, 0.0, 1.0)
a_msprt28 = (p_av[:, 27] < ALPHA).mean()
a_msprt60 = (p_av[:, 59] < ALPHA).mean()

# граница в z-шкале: какой |z| нужен для Lambda = 1/alpha в день k
from scipy.optimize import brentq

bound_z = []
for k in range(DAYS):
    vk = V_K[k]
    f = lambda zz: msprt_loglr(zz * np.sqrt(vk), vk) - np.log(1 / ALPHA)
    bound_z.append(brentq(f, 0.1, 50))
bound_z = np.array(bound_z)

print("=" * 88)
print("2) Упрощённый mSPRT (смесь N(0, tau^2), tau = 0,1 сигмы), те же A/A-миры:")
print(f"   наивный min p за 28 дней: {a_daily28:.1%}; за 60: {a_daily60:.1%}")
print(f"   always-valid p < 0,05 за 28 дней: {a_msprt28:.1%}; за 60 дней: {a_msprt60:.1%}")
print(f"   граница |z| для решимости: день 1 = {bound_z[0]:.2f}, день 7 = {bound_z[6]:.2f},"
      f" день 28 = {bound_z[27]:.2f}, день 60 = {bound_z[59]:.2f} (фиксированный порог 1,96)")
print("   Граница в z-шкале сначала снижается, затем растёт ~sqrt(ln n) —")
print("   это и есть 'размазывание' альфы по времени вместо одноразовой траты.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
show = np.arange(0, 400)
for i in show:
    axes[0].plot(K_DAYS[:40], p0[i, :40], lw=0.4, alpha=0.35, color="#C44E52",
                 label="наивный p (каждый день)" if i == show[0] else None)
    axes[0].plot(K_DAYS[:40], p_av[i, :40], lw=0.4, alpha=0.35, color="#4C72B0",
                 label="always-valid p (mSPRT)" if i == show[0] else None)
axes[0].axhline(ALPHA, color="black", ls="--", lw=1.6)
axes[0].set_yscale("log"); axes[0].set_ylim(1e-3, 1.5)
axes[0].set_xlabel("день A/A-теста"); axes[0].set_ylabel("p-value (log)")
axes[0].set_title(f"A/A, 400 миров: наивный p ныряет под 0,05 в {a_daily28:.0%} миров,\n"
                  f"always-valid — в {a_msprt60:.1%} за 60 дней")
axes[0].legend(fontsize=9, loc="lower left")

for i in np.arange(0, 60):
    axes[1].plot(K_DAYS[:40], z0[i, :40], lw=0.7, alpha=0.6,
                 color="#C44E52" if np.any(np.abs(z0[i, :40]) > bound_z[:40]) else "#4C72B0")
axes[1].plot(K_DAYS[:40], bound_z[:40], lw=2.6, color="#DD8452", label="mSPRT-граница")
axes[1].plot(K_DAYS[:40], -bound_z[:40], lw=2.6, color="#DD8452")
axes[1].axhline(1.96, color="gray", ls="--", lw=1.4, label="фиксированный порог 1,96")
axes[1].axhline(-1.96, color="gray", ls="--", lw=1.4)
axes[1].set_xlabel("день теста"); axes[1].set_ylabel("накопленная z-статистика")
axes[1].set_title("Normal-mixture граница: начальное снижение, затем рост")
axes[1].legend(fontsize=9, loc="lower right")
fig.suptitle("mSPRT: можно смотреть каждый день — уровень держит mixture likelihood ratio",
             fontsize=11.5, fontweight="bold")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_3_7_msprt.png", dpi=150)
print("Сохранено: practice_3_7_msprt.png")

# %% [markdown]
# ## Шаг 3. Цена и выгода: дни до решения при фиксированном дизайне
# Честное сравнение: fixed-horizon СПРОЕКТИРОВАН под MDE = 0,05 сигмы
# (мощность 80% -> 32 дня), решение строго в последний день. Sequential
# (ежедневный mSPRT) не знает истинный эффект. Смотрим три сценария:
# эффект = MDE, 2*MDE, 4*MDE (и симметричный вред -2*MDE для guardrails).

# %%
Z_A, Z_B = 1.96, 0.84
MDE = 0.05
D_FIX = int(np.ceil(2 * (Z_A + Z_B) ** 2 / MDE**2 / N_DAY))  # фиксированный план, дней


def sequential_days(delta, worlds=10_000):
    x, _, _ = simulate_paths(RNG, effect=delta, worlds=worlds)
    llr = msprt_loglr(x, V_K)
    cross = np.maximum.accumulate(llr, axis=1) > np.log(1 / ALPHA)
    detected = cross.any(axis=1)
    first = np.where(detected, cross.argmax(axis=1) + 1, DAYS)
    return detected, first


rows = []
for label, delta in [("= MDE (0,05σ)", MDE), ("2×MDE (0,10σ)", 2 * MDE), ("4×MDE (0,20σ)", 4 * MDE)]:
    _, z_h1, _ = simulate_paths(RNG, effect=delta, worlds=10_000)
    power_fix = (np.abs(z_h1[:, D_FIX - 1]) > Z_A).mean()   # мощность фикса в его день
    det, first = sequential_days(delta)
    rows.append({
        "истинный эффект": label,
        "fixed: дней (план)": D_FIX,
        "fixed: детект в день 32": f"{power_fix:.0%}",
        "seq: детект за 60 дн": f"{det.mean():.0%}",
        "seq: медиана дней": int(np.median(first)),
        "seq: среднее дней": f"{first.mean():.1f}",
    })

det_harm, first_harm = sequential_days(-2 * MDE)  # guardrail: ВРЕД 2xMDE

tab = pd.DataFrame(rows)
print("=" * 88)
print(f"3) Дни до решения, 200 юзеров/день на руку, alpha = 5%; fixed спроектирован под MDE = {MDE}σ = {D_FIX} дней:")
print(tab.to_string(index=False))
print(f"   Вред -2xMDE (guardrail): mSPRT ловит его за медиану {int(np.median(first_harm))} дн."
      f" (детект {det_harm.mean():.0%} за 60 дн) — сильные сигналы пробивают границу рано.")
print("   Итог: эффект = MDE -> sequential примерно как fixed (цена гибкости ~несколько дней);")
print("   эффект > MDE -> sequential останавливается в разы раньше; и в любом сценарии")
print("   уровень 5% держится при ежедневных проверках — в отличие от наивного p<0,05.")

# %%
fig, ax = plt.subplots(figsize=(8.6, 4.4))
labels = [r["истинный эффект"] for r in rows]
fix_d = [D_FIX] * len(rows)
seq_d = [r["seq: медиана дней"] for r in rows]
xb = np.arange(len(rows))
ax.bar(xb - 0.2, fix_d, width=0.38, color="#4C72B0", label=f"fixed-horizon: всегда {D_FIX} дн. (план под MDE)")
ax.bar(xb + 0.2, seq_d, width=0.38, color="#55A868", label="sequential mSPRT (медиана дней)")
for i, (f, s) in enumerate(zip(fix_d, seq_d)):
    ax.text(i - 0.2, f + 0.4, f"{f} дн.", ha="center", fontweight="bold")
    ax.text(i + 0.2, s + 0.4, f"{s} дн.", ha="center", fontweight="bold")
ax.set_xticks(xb); ax.set_xticklabels(labels)
ax.set_ylabel("дней до решения")
ax.set_title("Гибкость почти бесплатна на MDE и выгодна в разы на больших эффектах")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_3_7_speed.png", dpi=150)
print("Сохранено: practice_3_7_speed.png")

# %%
print()
print("=" * 88)
print("ГЛАВНОЕ (что должны увидеть):")
print(f"1) Подглядывание: один взгляд = {a_fixed28:.1%}, ежедневный (28 дн.) = {a_daily28:.0%},"
      f" 60 дней = {a_daily60:.0%} ложных побед.")
print(f"   Продление после взгляда (14->42) даёт {a_extend:.1%} — альфа уже потрачена, назад не откатить.")
print(f"2) Прокрас на 14-й день доживает до 28-го только в {confirm_share:.0%} случаев; ранний")
print("   прокрас преувеличивает эффект — останавливаться по нему нельзя (и без поправки).")
print(f"3) mSPRT: always-valid p держит A/A-уровень ({a_msprt60:.1%} за 60 дней, неравенство Вилля)")
print(f"   при ежедневных проверках; граница |z| растёт: {bound_z[0]:.2f} (день 1) -> {bound_z[27]:.2f} (день 28).")
print(f"4) Скорость (fixed спроектирован под MDE={MDE}σ, {D_FIX} дн.): эффект = MDE -> sequential ~ паритет")
print(f"   (медиана {rows[0]['seq: медиана дней']} дн.); 2xMDE -> {rows[1]['seq: медиана дней']} дн.; 4xMDE -> {rows[2]['seq: медиана дней']} дн.;")
print(f"   вред -2xMDE ловится за медиану {int(np.median(first_harm))} дн. — гибкость покупается")
print("   консервативностью на MDE, а не бесплатна.")
