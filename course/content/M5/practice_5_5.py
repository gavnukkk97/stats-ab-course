# -*- coding: utf-8 -*-
"""Практика 5.5 — Иерархические модели: partial pooling и MCMC (кейс «ЕдаДома»).

Финал модуля: города перестают врать поодиночке.

Что делаем:
1) конверсия по 12 городам «ЕдаДома»: два вруна — no pooling (Норильск 0/40
   = «конверсия 0%») и complete pooling («одна конверсия на все города»);
2) иерархическая бета-биномиальная модель в PyMC:
   k_c ~ Binomial(n_c, p_c),  p_c ~ Beta(mu*kappa, (1-mu)*kappa)
   — города тянутся к общему mu силой, обратно пропорциональной своему n
   (частичное заимствование, shrinkage);
3) сравниваем raw / pooled / shrunked оценки + веса w = n/(n+kappa);
4) диагностика MCMC: trace plot, R-hat, ESS (ArviZ);
5) иерархия в АБ: глобальный эффект + эффекты по городам
   vs «нарезка сегментов + поправка Холма» (урок 4.4) на одном сценарии.

Запуск:  python3 practice_5_5.py   (нужен pymc:  pip3 install --user pymc arviz)
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# # Практика 5.5 — Города, которые тянутся к среднему
# До сих пор (5.1–5.3) апостериор считался формулой: сопряжённость
# Beta-Бернулли закрывала вопрос. Сейчас параметров 14 (mu, kappa + 12
# конверсий городов), аналитической формулы нет — зовём MCMC (PyMC).
#
# Техническое примечание: arviz 0.17 (последний для python 3.9) ждёт
# scipy < 1.13 и падает на импорте scipy.signal.gaussian. Чиним шимом
# ДО импорта pymc — сам shim в вычислениях не участвует.

# %%
import warnings

import scipy.signal as _sig

if not hasattr(_sig, "gaussian"):  # shim: scipy>=1.13 убрал gaussian из scipy.signal
    from scipy.signal.windows import gaussian as _gaussian

    _sig.gaussian = _gaussian

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats

RNG = np.random.default_rng(55)  # seed -> результат воспроизводим
HERE = Path(__file__).resolve().parent
ALPHA = 0.05

print(f"PyMC {pm.__version__} | ArviZ {az.__version__}")


def holm(pvals):
    """Поправка Холма (контроль FWER): скорректированные p и флаги отвержения."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m, run, adj = len(p), 0.0, np.empty(len(p))
    for r, idx in enumerate(order):
        run = max(run, (m - r) * p[idx])
        adj[idx] = min(1.0, run)
    return adj, adj < ALPHA


# %% [markdown]
# ## Шаг 1. Данные: конверсия показа в первый заказ по 12 городам
# Месячная выгрузка из админки. Два города-мучителя: Сыктывкар (38 из 80 —
# 47,5%!) и Норильск (0 из 40 — 0%). Обе цифры — шум на крошечном n,
# но в отчёте они выглядят как «лучший» и «худший» город страны.

# %%
cities = pd.DataFrame(
    {
        "city": [
            "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань",
            "Нижний Новгород", "Челябинск", "Самара", "Омск", "Ростов-на-Дону",
            "Сыктывкар", "Норильск",
        ],
        "n": np.array([52_000, 31_000, 12_000, 9_500, 7_000, 5_500, 4_000, 2_800, 1_600, 900, 80, 40]),
        "k": np.array([15_812, 9_301, 3_686, 2_916, 2_163, 1_668, 1_196, 874, 462, 258, 38, 0]),
    }
)
C = len(cities)
n_c = cities["n"].to_numpy()
k_c = cities["k"].to_numpy()
cities["raw, %"] = k_c / n_c * 100
pooled = k_c.sum() / n_c.sum()  # complete pooling: одна конверсия на всех
cities["pooled, %"] = pooled * 100

print("=" * 96)
print("1) Конверсия по городам: raw (no pooling) vs pooled (complete pooling)")
print(cities.set_index("city")[["n", "k", "raw, %", "pooled, %"]].round(1).to_string())
print(f"   Полная подушка: {pooled:.4f}. Норильск raw 0% (CI Уилсона 0–9%), Сыктывкар raw 47,5% —")
print("   оба врут: на n=40 биномиальный шум размаха десятки п.п. (закон малых чисел, урок 1.1).")

# %% [markdown]
# ## Шаг 2. Иерархическая модель и MCMC
# k_c ~ Binomial(n_c, p_c);  p_c ~ Beta(mu*kappa, (1-mu)*kappa).
# Гиперпараметры тоже случайные: mu — «конверсия среднего города»,
# kappa — «сила» общего распределения, измеряется в юзерах.
# Сопряжённости на 14-мерном апостериоре нет -> Metropolis/NUTS внутри pm.sample.

# %%
with pm.Model() as m_hier:
    mu = pm.Beta("mu", alpha=2, beta=5)                  # глобальная конверсия
    kappa = pm.Gamma("kappa", alpha=2, beta=0.01)        # концентрация (в юзерах)
    p_city = pm.Beta("p_city", mu * kappa, (1 - mu) * kappa, shape=C)
    obs = pm.Binomial("k", n=n_c, p=p_city, observed=k_c)
    idata = pm.sample(2_000, tune=2_000, chains=2, cores=1, target_accept=0.95,
                      random_seed=55, progressbar=False)

summ = az.summary(idata, var_names=["mu", "kappa", "p_city"], round_to=4)
rhat_max = summ["r_hat"].max()
ess_min = summ["ess_bulk"].min()

print("=" * 96)
print("2) Диагностика MCMC (ArviZ)")
print(f"   R-hat max = {rhat_max:.4f} (норма <= 1.01: цепи согласны между собой)")
print(f"   ESS bulk min = {ess_min:.0f} (эффективный размер выборки; автокорреляция цепи его съедает)")
print(f"   mu (глобальная конверсия): {summ.loc['mu', 'mean']:.4f}  "
      f"94% HDI [{summ.loc['mu', 'hdi_3%']:.4f}; {summ.loc['mu', 'hdi_97%']:.4f}]")
print(f"   kappa (сила сжатия, в юзерах): {summ.loc['kappa', 'mean']:.0f}  "
      f"94% HDI [{summ.loc['kappa', 'hdi_3%']:.0f}; {summ.loc['kappa', 'hdi_97%']:.0f}]")

# trace plot: «хвост-гусеница» без трендов = здоровая цепь
axes_tr = az.plot_trace(idata, var_names=["mu", "kappa"], figsize=(9, 4.4), compact=True)
axes_tr[0, 0].set_title("mu: след цепей (нужен «хвост-гусеница»)")
axes_tr[1, 0].set_title("kappa: след цепей")
axes_tr[0, 1].set_title("")
axes_tr[1, 1].set_title("")
fig_tr = axes_tr[0, 0].figure
fig_tr.tight_layout()
fig_tr.savefig(HERE / "practice_5_5_trace.png", dpi=150, bbox_inches="tight")
print("   Сохранено: practice_5_5_trace.png")

# %% [markdown]
# ## Шаг 3. Raw vs shrunked: частичное заимствование в числах
# Апостериорное среднее города ~ взвешенное среднее своего raw и общего mu:
# E[p_c] ~ (k + mu*kappa)/(n + kappa) = w*raw + (1-w)*mu,  w = n/(n+kappa).
# Сравним аналитическую формулу (по posterior-mean mu и kappa) с MCMC.

# %%
mu_hat = float(summ.loc["mu", "mean"])
kap_hat = float(summ.loc["kappa", "mean"])
shrunk_mcmc = summ.loc[[f"p_city[{i}]" for i in range(C)], "mean"].to_numpy()
hdi_lo = summ.loc[[f"p_city[{i}]" for i in range(C)], "hdi_3%"].to_numpy()
hdi_hi = summ.loc[[f"p_city[{i}]" for i in range(C)], "hdi_97%"].to_numpy()

cities["shrunk, %"] = shrunk_mcmc * 100
cities["вес своего w"] = n_c / (n_c + kap_hat)
cities["формула, %"] = (cities["вес своего w"] * k_c / n_c + (1 - cities["вес своего w"]) * mu_hat) * 100
cities["HDI низ, %"] = hdi_lo * 100
cities["HDI верх, %"] = hdi_hi * 100

print("=" * 96)
print("3) Три оценки конверсии: raw / pooled / shrunked (MCMC) + аналитическая формула")
out = cities.set_index("city")[["n", "raw, %", "shrunk, %", "формула, %", "вес своего w", "HDI низ, %", "HDI верх, %"]]
print(out.round(3).to_string())
print(f"   Формула w*raw+(1-w)*mu воспроизводит MCMC: макс. расхождение "
      f"{np.abs(cities['shrunk, %'] - cities['формула, %']).max():.2f} п.п.")
print()
nor = cities.set_index("city").loc["Норильск"]
syr = cities.set_index("city").loc["Сыктывкар"]
msk = cities.set_index("city").loc["Москва"]
print(f"   Норильск: raw {nor['raw, %']:.0f}% -> shrunk {nor['shrunk, %']:.1f}% "
      f"(вес своего опыта w = {nor['вес своего w']:.3f} — почти весь город из общего)")
print(f"   Сыктывкар: raw {syr['raw, %']:.0f}% -> shrunk {syr['shrunk, %']:.1f}% — сжатие работает в обе стороны")
print(f"   Москва: raw {msk['raw, %']:.1f}% -> shrunk {msk['shrunk, %']:.1f}% "
      f"(w = {msk['вес своего w']:.3f} — большой город говорит сам за себя)")

# shrinkage-график: raw -> shrunked стрелками, цвет = размер города
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
order = np.argsort(-n_c)  # слева большие города, справа маленькие
yy_raw = cities["raw, %"].to_numpy()[order]
yy_shr = cities["shrunk, %"].to_numpy()[order]
names_o = cities["city"].to_numpy()[order]
n_o = n_c[order]
sc = axes[0].scatter(range(C), yy_raw, c=np.log10(n_o), cmap="viridis", s=70, zorder=3, label="raw (no pooling)")
axes[0].scatter(range(C), yy_shr, c=np.log10(n_o), cmap="viridis", s=70, marker="s", zorder=3,
                edgecolor="black", linewidth=0.6, label="shrunked (иерархия)")
for i in range(C):
    axes[0].annotate("", xy=(i, yy_shr[i]), xytext=(i, yy_raw[i]),
                     arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.4, alpha=0.8))
axes[0].axhline(pooled * 100, color="black", ls="--", lw=1.5, label=f"pooled {pooled*100:.1f}%")
axes[0].set_xticks(range(C), names_o, rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("конверсия показа в заказ, %")
axes[0].set_title("raw (кружки) -> shrunked (квадраты)", fontsize=10)
axes[0].legend(fontsize=8)
plt.colorbar(sc, ax=axes[0], label="log10(n юзеров)")

nn = np.logspace(1, 5.3, 200)
axes[1].plot(nn, nn / (nn + kap_hat), color="#4C72B0", lw=2,
             label=f"w = n/(n+kappa), kappa = {kap_hat:.0f}")
axes[1].scatter(n_c, cities["вес своего w"], color="#C44E52", zorder=3, s=28)
for name, x, yw in [("Норильск", 40, float(nor["вес своего w"])),
                    ("Сыктывкар", 80, float(syr["вес своего w"])),
                    ("Москва", 52_000, float(msk["вес своего w"]))]:
    axes[1].annotate(name, (x, yw), xytext=(6, -10), textcoords="offset points", fontsize=8)
axes[1].axhline(0.5, color="grey", ls=":", lw=1.2)
axes[1].set_xscale("log")
axes[1].set_xlabel("n юзеров в городе (log)")
axes[1].set_ylabel("вес собственного опыта w")
axes[1].set_title("вес своего опыта w = n/(n+kappa)", fontsize=10)
axes[1].legend(fontsize=9)
fig.suptitle("Частичное заимствование: малые города тянутся к общему сильнее больших; kappa измеряется в юзерах",
             y=1.00, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(HERE / "practice_5_5_shrinkage.png", dpi=150, bbox_inches="tight")
print("   Сохранено: practice_5_5_shrinkage.png")

# %% [markdown]
# ## Шаг 4. Иерархия в АБ: глобальный эффект vs «нарезка + Холм» (4.4)
# Сценарий: тест «промо-баннер на главном экране». Истинный глобальный
# аплифт +0,8 п.п. (база ~30%), по городам N(+0,8; 0,6) п.п., сплит 50/50.
# Частотный путь из 4.4: 12 отдельных z-тестов + поправка Холма.
# Иерархический путь: d_city ~ Normal(delta, tau), delta — один глобальный
# параметр (поправки за множественность нет — гипотеза одна).

# %%
base_true = RNG.normal(0.30, 0.010, C)              # истинные конверсии городов
uplift_true = RNG.normal(0.010, 0.006, C)           # истинные аплифты по городам: глобальный +1,0 п.п.
n_t, n_ctl = n_c // 2, n_c - n_c // 2
k_t = RNG.binomial(n_t, np.clip(base_true + uplift_true, 0.01, 0.98))
k_ctl = RNG.binomial(n_ctl, base_true)

rows = []
for i in range(C):
    p1, p2 = k_t[i] / n_t[i], k_ctl[i] / n_ctl[i]
    pbar = (k_t[i] + k_ctl[i]) / (n_t[i] + n_ctl[i])
    se = np.sqrt(pbar * (1 - pbar) * (1 / n_t[i] + 1 / n_ctl[i]))
    z = (p1 - p2) / se
    rows.append({"city": cities["city"][i], "эффект, п.п.": (p1 - p2) * 100,
                 "истина, п.п.": uplift_true[i] * 100, "p сырой": 2 * stats.norm.sf(abs(z)), "se": se})
ab = pd.DataFrame(rows)
adj, rej = holm(ab["p сырой"].values)
ab["p Холм"] = adj
ab["значим (Холм)"] = np.where(rej, "да", "нет")

best_p = ab.loc[ab["p сырой"].idxmin()]
best_eff = ab.loc[ab["эффект, п.п."].idxmax()]
n_raw_sig = (ab["p сырой"] < ALPHA).sum()

# pooled-тест на всей выборке (complete pooling эффекта) — для честности
p1a, p2a = k_t.sum() / n_t.sum(), k_ctl.sum() / n_ctl.sum()
pbar_a = (k_t.sum() + k_ctl.sum()) / (n_t.sum() + n_ctl.sum())
z_pool = (p1a - p2a) / np.sqrt(pbar_a * (1 - pbar_a) * (1 / n_t.sum() + 1 / n_ctl.sum()))

print("=" * 96)
print("4а) «Нарезка сегментов + Холм» (урок 4.4): 12 отдельных z-тестов")
print(ab.set_index("city").round(3).drop(columns="se").to_string())
print(f"   Сырых «значимых» городов: {n_raw_sig} из 12; после Холма: {rej.sum()}.")
print(f"   Минимальный p: {best_p['city']} ({best_p['p сырой']:.3f}) -> Холм {best_p['p Холм']:.2f} — не значимо.")
print(f"   Максимальный эффект: {best_eff['city']}: {best_eff['эффект, п.п.']:+.1f} п.п. при истине "
      f"{best_eff['истина, п.п.']:+.1f} п.п. — классический winner's curse:")
print("   максимум по шумным оценкам всегда оптимистичен (уроки 4.4–4.5).")
print(f"   (Справки ради: pooled z-тест на всей выборке: {z_pool:.2f}, "
      f"p = {2*stats.norm.sf(abs(z_pool)):.4f} — один тест без нарезки мощнее нарезки с Холмом.)")

# %% [markdown]
# Иерархическая модель на оценках эффектов (нормальная аппроксимация —
# классическая схема «8 школ»): y_c ~ Normal(d_c, se_c), d_c ~ Normal(delta, tau).
# Параметризация нецентрированная (d = delta + tau*z, z ~ N(0,1)): центрированная
# ловит «воронку» при малых tau — дивергенции и R-hat > 1,01 (Мартин, гл. 6).

# %%
est = (k_t / n_t - k_ctl / n_ctl)
se = ab["se"].to_numpy() * 100  # в п.п.
with pm.Model() as m_ab:
    delta = pm.Normal("delta", mu=0, sigma=3.0)        # глобальный эффект, п.п.
    tau = pm.HalfNormal("tau", sigma=2.0)              # разброс эффектов между городами
    z = pm.Normal("z", 0.0, 1.0, shape=C)              # нецентрированная параметризация
    d_city = pm.Deterministic("d_city", delta + tau * z)
    y = pm.Normal("y_obs", mu=d_city, sigma=se, observed=est * 100)
    idata_ab = pm.sample(2_000, tune=2_000, chains=2, cores=1, target_accept=0.95,
                         random_seed=55, progressbar=False)

summ_ab = az.summary(idata_ab, var_names=["delta", "tau", "d_city"], round_to=4)
delta_mean = float(summ_ab.loc["delta", "mean"])
d_lo, d_hi = float(summ_ab.loc["delta", "hdi_3%"]), float(summ_ab.loc["delta", "hdi_97%"])
p_pos = float((idata_ab.posterior["delta"] > 0).mean())
d_shrunk = summ_ab.loc[[f"d_city[{i}]" for i in range(C)], "mean"].to_numpy()
rhat_ab = float(summ_ab["r_hat"].max())

print("=" * 96)
print("4б) Иерархическая модель: один глобальный delta + сжатые эффекты городов")
print(f"   R-hat max = {rhat_ab:.4f}")
print(f"   Глобальный эффект delta = {delta_mean:+.2f} п.п.  94% HDI [{d_lo:+.2f}; {d_hi:+.2f}]  "
      f"P(delta>0) = {p_pos:.3f}")
print(f"   (истина: +1,00 п.п.; tau posterior mean = {float(summ_ab.loc['tau', 'mean']):.2f} п.п.)")
cmp_tab = pd.DataFrame({"city": cities["city"], "эффект raw, п.п.": ab["эффект, п.п."].to_numpy(),
                        "эффект shrunk, п.п.": d_shrunk, "истина, п.п.": uplift_true * 100})
print(cmp_tab.set_index("city").round(2).to_string())
err_shr = np.abs(cmp_tab["эффект shrunk, п.п."] - cmp_tab["истина, п.п."]).mean()
err_raw = np.abs(cmp_tab["эффект raw, п.п."] - cmp_tab["истина, п.п."]).mean()
print(f"   Средняя ошибка оценки эффекта города: raw {err_raw:.2f} п.п. -> shrunk {err_shr:.2f} п.п. "
      f"({err_raw/err_shr:.1f}x точнее) — вот зачем частичное заимствование.")

# график: raw CI vs shrunk + глобальный delta
fig, ax = plt.subplots(figsize=(12.0, 5.2))
order = np.argsort(-n_c)
x = np.arange(C)
raw_o = ab["эффект, п.п."].to_numpy()[order]
shr_o = d_shrunk[order]
true_o = uplift_true[order] * 100
n_o = n_c[order]
err = 1.96 * se[order]
ax.errorbar(x - 0.13, raw_o, yerr=err, fmt="o", color="#4C72B0", lw=1.4, capsize=2.5, label="raw эффект + 95% CI (частотный)")
ax.scatter(x + 0.13, shr_o, marker="s", color="#55A868", zorder=3, label="shrunked (иерархия)")
ax.scatter(x, true_o, marker="x", color="black", zorder=3, s=42, label="истинный эффект (из генератора)")
ax.axhspan(d_lo, d_hi, color="#55A868", alpha=0.18, label=f"глобальный delta 94% HDI [{d_lo:+.1f}; {d_hi:+.1f}] п.п.")
ax.axhline(delta_mean, color="#55A868", ls="--", lw=1.6)
ax.axhline(0, color="grey", lw=1)
for i in range(C):
    if rej[order][i]:
        ax.annotate("Холм: значимо", (i, raw_o[i]), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=7, color="#C44E52")
ax.set_xticks(x, [f"{c}\nn={n:,}".replace(",", " ") for c, n in zip(cities["city"][order], n_o)],
              rotation=45, ha="right", fontsize=8)
ax.set_ylabel("эффект на конверсию, п.п.")
ax.set_title(f"АБ по городам: после Холма значимых {rej.sum()} из 12 — а иерархия ловит глобальный эффект\n"
             f"{delta_mean:+.1f} п.п. (P(delta>0) = {p_pos:.2f}) и сжимает шумные городские оценки к нему (x — истина)",
             fontsize=11)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(HERE / "practice_5_5_ab_segments.png", dpi=150, bbox_inches="tight")
print("   Сохранено: practice_5_5_ab_segments.png")

print("=" * 96)
print("ИТОГ:")
print(f"1) No pooling врёт шумом (Норильск 0%, Сыктывкар 47,5%), complete pooling — усреднением.")
print(f"2) Иерархия: kappa = {kap_hat:.0f} юзеров; shrunk-оценки = w*raw + (1-w)*mu, w = n/(n+kappa).")
print(f"3) Нарезка+Холм: {rej.sum()} значимых городов; иерархия: delta = {delta_mean:+.2f} п.п., "
      f"P(delta>0) = {p_pos:.3f} — одна гипотеза, без штрафа за множественность.")
print(f"4) Сжатие чинит winner's curse: сырой максимум {best_eff['эффект, п.п.']:+.1f} п.п. -> "
      f"shrunk {d_shrunk[int(ab['эффект, п.п.'].idxmax())]:+.1f} п.п. при истине {best_eff['истина, п.п.']:+.1f} п.п.")
