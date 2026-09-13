# -*- coding: utf-8 -*-
"""Практика 6.3 — Квазиэксперименты: DiD, staggered, synthetic control, DiD+PSM.

Кейс «ЕдаДома». Не верь — проверь симуляцией: строим миры, где истинный эффект
нам известен, и смотрим, кто из оценщиков его находит, а кто врёт.

Блоки:
1) DiD 2×2 руками + регрессией (numpy lstsq с дамми). Наивный pre-post против
   DiD при наличии общего тренда: тренд достаётся контролю, эффект — β3.
2) Pretrend-тест: event study с 95% CI на данных БЕЗ эффекта — все коэффициенты
   должны быть нулями; плюс плацебо-период (фейковая дата лечения на пре-данных).
3) Ловушка staggered TWFE: 3 когорты лечения + чистый контроль, эффекты
   неоднородны и нарастают; наивный TWFE (уже-лечёные как контроли) врёт,
   групповой DiD «когорта против никогда-не-лечёных» чинит.
4) Synthetic control: 1 treated-город + 9 контрольных, веса lstsq по пре-периоду
   (проекция на симплекс), разрыв «факт − синтетик» = эффект; плацебо-юниты
   (TRoP-логика: пермутационный тест по ratio post/pre MSPE).
5) Мини-кейс DiD+PSM: панель 1500 курьеров, 6 периодов, конфаундер в назначении
   на тренинг, истинный tau = 500. Лестница: naive DiD -> DiD + ковариаты
   (аддитивно и в тренд) -> matching по propensity (свой logit-IRLS + ближайший
   сосед с калипером) -> DiD на отматченных. Грабля: матчинг с post-treatment
   ковариатой (рейтинг качества в первый пост-месяц, на него тренинг уже
   повлиял) смещает оценку.

Запуск:  python3 practice_6_3.py
Графики (PNG) сохраняются рядом со скриптом. Статистика — numpy/scipy/pandas,
OLS везде через np.linalg.lstsq (без statsmodels).
"""

# %% [markdown]
# # Практика 6.3 — Квазиэксперименты
# Правило урока: у квазиэксперимента нет рандомизации, поэтому «контроль» надо
# КОНСТРУИРОВАТЬ (другой город, двойники, синтетик, порог) — и отдельно
# доказывать, что он годится. Истина известна только в симуляции — проверяем всех.

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(63)  # seed -> результат воспроизводим
HERE = Path(__file__).resolve().parent
ALPHA = 0.05


def ols(y, X):
    """OLS через numpy.linalg.lstsq: beta, SE, t, p (классическая ковариация)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(X.shape[0] - np.linalg.matrix_rank(X), 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
        p = 2 * stats.t.sf(np.abs(t), dof)
    return beta, se, t, p


print("=" * 100)
print("ПРАКТИКА 6.3 — Квазиэксперименты (ЕдаДома): DiD / staggered / synth / DiD+PSM")
print("=" * 100)

# %% [markdown]
# ## Блок 1. DiD 2×2: руками и регрессией, против наивного pre-post
# «ЕдаДома» запускает подписку «Прайм» (бесплатная доставка) в одном городе.
# Общего тренда не избежать: бизнес растёт везде (~60 ₽/мес к чеку + сезонность).
# Эффект подписки +250 ₽. Наивный pre-post припишет подписке ещё и тренд.

# %%
N_USERS, T_PRE, T_POST, TREND, TAU = 300, 4, 4, 60.0, 250.0
periods = np.arange(1, T_PRE + T_POST + 1)
post1 = (periods > T_PRE).astype(int)
common = TREND * periods + 10 * np.sin(periods)     # общий тренд + сезонность
shock = np.where(periods > T_PRE, TAU, 0.0)         # эффект Прайма

rows = []
for u in range(N_USERS):
    base_c = RNG.normal(2000, 200)                  # контроль: свой уровень
    base_t = RNG.normal(2150, 200)                  # тест: уровень ВЫШЕ (селекция)
    eps = RNG.normal(0, 90, len(periods))
    for j, t in enumerate(periods):
        rows.append((2 * u, "контроль", t, post1[j], base_c + common[j] + eps[j]))
        rows.append((2 * u + 1, "тест", t, post1[j], base_t + common[j] + shock[j] + eps[j]))

df1 = pd.DataFrame(rows, columns=["user", "city", "t", "post", "y"])
m = df1.groupby(["city", "post"])["y"].mean().unstack()
pre_c, post_c = m.loc["контроль", 0], m.loc["контроль", 1]
pre_t, post_t = m.loc["тест", 0], m.loc["тест", 1]
naive_pp = post_t - pre_t
did_hand = (post_t - pre_t) - (post_c - pre_c)

# регрессия: y = b0 + b1*Post + b2*Treat + b3*Post*Treat
treat1 = (df1["city"] == "тест").astype(float).values
X1r = np.column_stack([np.ones(len(df1)), df1["post"].values, treat1,
                       df1["post"].values * treat1])
b1r, s1r, t1r, p1r = ols(df1["y"].values, X1r)

print("\n1) DiD 2×2 (подписка Прайм, истинный эффект = 250 ₽)")
print(pd.DataFrame({
    "До (мес 1–4)": [pre_c, pre_t],
    "После (мес 5–8)": [post_c, post_t],
    "Изменение": [post_c - pre_c, post_t - pre_t],
}, index=["Контроль", "Тест"]).round(1).to_string())
print(f"\n  Наивный pre-post (только тест):      {naive_pp:7.1f} ₽ ({naive_pp / pre_t * 100:+.1f}%)"
      "  — тренд+сезон приписаны подписке")
print(f"  DiD руками (разница разниц):        {did_hand:7.1f} ₽ ({did_hand / pre_t * 100:+.1f}%)")
print(f"  DiD регрессией, β3 (Post×Treat):    {b1r[3]:7.1f} ₽  SE={s1r[3]:.1f}, p={p1r[3]:.2g}  (истина 250)")
print(f"  β1 (Post) = {b1r[1]:.1f} ₽ — общий тренд; β2 (Treat) = {b1r[2]:.1f} ₽ — сдвиг уровней; "
      "DiD вычитает их сам")

g = df1.groupby(["t", "city"])["y"].mean().unstack()
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(g.index, g["контроль"], "o-", color="#4477aa", label="контроль (другие города)")
ax.plot(g.index, g["тест"], "o-", color="#ee6677", label="тест (город с Праймом)")
cf = g["тест"].loc[T_PRE] + (g["контроль"].loc[T_PRE + 1:].values - g["контроль"].loc[T_PRE])
ax.plot(range(T_PRE + 1, T_PRE + T_POST + 1), cf, "--", color="#ee6677", alpha=0.7,
        label="контрфакт: тест без Прайма (тренд как у контроля)")
ax.axvline(T_PRE + 0.5, color="gray", ls=":", lw=1)
ax.annotate("эффект = β3 (разрыв)", xy=(8, 0.5 * (g['тест'].loc[8] + cf[-1])),
            xytext=(5.2, cf[-1] - 300), arrowprops=dict(arrowstyle="->", color="k"), fontsize=9)
ax.set_title("DiD: тренд достаётся контролю, эффект — вертикальному разрыву")
ax.set_xlabel("месяц"); ax.set_ylabel("средняя выручка на юзера, ₽")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE / "practice_6_3_did.png", dpi=150); plt.close(fig)

# %% [markdown]
# ## Блок 2. Pretrend-тест: event study на данных БЕЗ эффекта
# Оцениваем «эффекты» в каждый месяц относительно последнего пре-периода
# (unit FE + time FE + Treat×месяц). Если параллельные тренды верны, все
# коэффициенты — и лиды до, и лаги после — неотличимы от нуля.
# Плюс плацебо: «лечение» в середине пре-данных — эффекта быть не должно.

# %%
N2, T2, G2 = 300, 10, 6   # «лечение» с периода 6, но эффект = 0 (плацебо-раскатка)
treat2 = np.repeat(np.array([0.0, 1.0]), N2 // 2)
mu2 = RNG.normal(50, 8, N2) + 20 * treat2           # уровни групп различаются — и не важно
time2 = 2.5 * np.arange(T2) + 8 * np.sin(np.arange(T2) * 0.9)
Y2 = mu2[:, None] + time2[None, :] + RNG.normal(0, 6, (N2, T2))

y2 = Y2.ravel(order="C")
cols = [np.ones(N2 * T2)]
cols += [np.repeat((np.arange(N2) == u).astype(float), T2) for u in range(1, N2)]  # unit FE
cols += [np.tile((np.arange(T2) == t).astype(float), N2) for t in range(T2) if t != G2 - 1]  # time FE
inter = [np.tile((np.arange(T2) == t).astype(float), N2) * np.repeat(treat2, T2)
         for t in range(T2) if t != G2 - 1]                                        # лиды/лаги
X2r = np.column_stack(cols + inter)
b2r, s2r, t2r, p2r = ols(y2, X2r)
coef_idx = 1 + (N2 - 1) + (T2 - 1) + np.arange(T2 - 1)

ev = np.delete(np.arange(T2) - (G2 - 1), G2 - 1)
ev_tab = pd.DataFrame({"месяц от лечения": ev, "оценка": b2r[coef_idx],
                       "SE": s2r[coef_idx]}).set_index("месяц от лечения")
ci_ok = int((((ev_tab["оценка"] - 1.96 * ev_tab["SE"] <= 0)
              & (ev_tab["оценка"] + 1.96 * ev_tab["SE"] >= 0)).sum()))

print("\n2) Event study на данных БЕЗ эффекта (лечение с периода 6, tau = 0)")
print(ev_tab.round(2).to_string())
print(f"  95% CI накрывает 0 у {ci_ok} из {len(ev_tab)} коэффициентов; "
      f"max |t| = {np.nanmax(np.abs(t2r[coef_idx])):.2f} — претрендов нет, и тест их не «находит»")

# плацебо-период: фейковое лечение на периоде 4, данные только 1–6 (всё — пре)
ypl = Y2[:, :6].ravel(order="C")
postpl = np.tile(np.arange(6) >= 3, N2).astype(float)
tpl = np.repeat(treat2, 6)
Xpl = np.column_stack([np.ones(N2 * 6), postpl, tpl, postpl * tpl])
bpl, spl, tpl2, ppl = ols(ypl, Xpl)
print(f"  Плацебо-период (фейковое лечение на мес 4, только пре-данные): "
      f"DiD = {bpl[3]:+6.2f}, p = {ppl[3]:.2f} -> ложный эффект не найден")

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(ev_tab.index, ev_tab["оценка"], yerr=1.96 * ev_tab["SE"], fmt="o",
            capsize=3, color="#4477aa", label="Treat×месяц (95% CI)")
ax.axhline(0, color="k", lw=1)
ax.axvline(-0.5, color="gray", ls=":", label="момент лечения (эффекта нет)")
ax.set_title("Event study без эффекта: и лиды, и лаги — нули")
ax.set_xlabel("месяцы от лечения (минус = до)"); ax.set_ylabel("оценка «эффекта»")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE / "practice_6_3_event_study.png", dpi=150); plt.close(fig)

# %% [markdown]
# ## Блок 3. Staggered adoption: ловушка TWFE
# «Прайм» раскатывается по городам волнами: когорты g = 3, 5, 7 + города, где не
# запускали (чистый контроль). Эффект неоднороден и НАРАСТАЕТ со стажем лечения:
# tau_it = 100 + 45*(t − g). Наивный TWFE разрешает сравнивать поздние когорты с
# ранними (уже лечёнными) — и вычитает из эффекта чужой накопленный эффект.

# %%
T3 = 10
cohorts3 = {3: 60, 5: 60, 7: 60, np.inf: 120}       # g -> число юнитов
G_LIST = [3, 5, 7]
offset3 = {3: 30, 5: 0, 7: -20, np.inf: 10}
n3 = sum(cohorts3.values())
g_unit = np.concatenate([np.full(ng, g) for g, ng in cohorts3.items()]).astype(float)
mu3 = np.concatenate([RNG.normal(100 + offset3[g], 6, ng) for g, ng in cohorts3.items()])
u3 = np.repeat(np.arange(n3), T3)
t3 = np.tile(np.arange(1, T3 + 1), n3)
gu, tt3 = g_unit[u3], t3.astype(float)
D_it = ((~np.isinf(gu)) & (tt3 >= gu)).astype(float)
tau_it = np.where(D_it > 0, 100 + 45 * (tt3 - gu), 0.0)
y3 = mu3[u3] + 10 * t3 + 3 * np.sin(t3) + tau_it + RNG.normal(0, 10, len(u3))
truth_overall = float(tau_it[D_it > 0].mean())

# наивный TWFE: unit FE + time FE + D_it
X3 = np.column_stack(
    [np.ones(n3 * T3)]
    + [(u3 == u).astype(float) for u in range(1, n3)]
    + [(t3 == t).astype(float) for t in range(2, T3 + 1)]
    + [D_it])
b3r, s3r, t3r, p3r = ols(y3, X3)

# групповой DiD (логика Callaway–Sant'Anna): ATT(g,t) против НИКОГДА не лечёных
Ymat = np.full((n3, T3 + 1), np.nan)
Ymat[u3, t3] = y3
never = np.isinf(g_unit)
rows_gt = []
for g in G_LIST:
    sel_g = g_unit == g
    for t in range(int(g), T3 + 1):
        att = ((Ymat[sel_g, t].mean() - Ymat[sel_g, int(g) - 1].mean())
               - (Ymat[never, t].mean() - Ymat[never, int(g) - 1].mean()))
        rows_gt.append({"g": g, "t": t, "ATT(g,t)": att, "истина": 100 + 45 * (t - g)})
gt = pd.DataFrame(rows_gt)
w_gt = [cohorts3[r["g"]] for _, r in gt.iterrows()]
est_group = float(np.average(gt["ATT(g,t)"], weights=w_gt))

# бутстреп SE группового DiD (ресемплинг юнитов внутри групп, 200 реп.)
idx_by = {g: np.where(g_unit == g)[0] for g in G_LIST + [np.inf]}
bs3 = []
for _ in range(200):
    samp = {g: RNG.choice(idx_by[g], cohorts3[g], replace=True) for g in G_LIST + [np.inf]}
    vals, wts = [], []
    for g in G_LIST:
        for t in range(int(g), T3 + 1):
            vals.append((Ymat[samp[g], t].mean() - Ymat[samp[g], int(g) - 1].mean())
                        - (Ymat[samp[np.inf], t].mean() - Ymat[samp[np.inf], int(g) - 1].mean()))
            wts.append(cohorts3[g])
    bs3.append(np.average(vals, weights=wts))
se_group = float(np.std(bs3, ddof=1))

print("\n3) Staggered adoption: наивный TWFE против группового DiD (эффекты нарастают)")
print(gt.pivot(index="t", columns="g", values=["ATT(g,t)", "истина"]).round(1).to_string())
print(f"\n  Истинный средний эффект (по лечёным наблюдениям): {truth_overall:6.1f}")
print(f"  Наивный TWFE (D_it, unit+time FE):           {b3r[-1]:6.1f} (SE={s3r[-1]:.1f})"
      f"  — смещение {100 * (b3r[-1] - truth_overall) / truth_overall:+.0f}%")
print(f"  Групповой DiD (когорта vs чистый контроль):  {est_group:6.1f} (SE={se_group:.1f}, бутстреп)"
      f"  — смещение {100 * (est_group - truth_overall) / truth_overall:+.0f}%")
print("  Механика провала: TWFE сравнивает поздние когорты и с ранними — уже лечёнными,")
print("  у которых к тому моменту накоплен свой эффект, — и вычитает его из оценки.")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
for g, c in zip(G_LIST, ["#ee6677", "#228833", "#aa3377"]):
    mns = [np.nanmean(Ymat[g_unit == g, t]) for t in range(1, T3 + 1)]
    ax.plot(range(1, T3 + 1), mns, "o-", ms=3, color=c, label=f"когорта g={g}")
mns = [np.nanmean(Ymat[never, t]) for t in range(1, T3 + 1)]
ax.plot(range(1, T3 + 1), mns, "o-", ms=3, color="#4477aa", label="никогда не лечёные")
ax.set_title("Когорты волнами: эффекты нарастают со стажем лечения")
ax.set_xlabel("период"); ax.set_ylabel("Y"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax = axes[1]
names = ["истина", "TWFE\n(наивный)", "групповой DiD\n(C&S-логика)"]
ax.bar(names, [truth_overall, b3r[-1], est_group],
       color=["#888888", "#cc3311", "#228833"], yerr=[0, s3r[-1], se_group], capsize=4)
for i, v in enumerate([truth_overall, b3r[-1], est_group]):
    ax.text(i, v + 10, f"{v:.0f}", ha="center", fontsize=10)
ax.set_title(f"TWFE занижает эффект на {abs(100 * (b3r[-1] - truth_overall) / truth_overall):.0f}%")
ax.set_ylabel("ATT"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(HERE / "practice_6_3_staggered.png", dpi=150); plt.close(fig)

# %% [markdown]
# ## Блок 4. Synthetic control: один город, синтетический двойник из девяти
# В городе 0 открыли хаб «ЕдаДома» (darkstore). A/B невозможен: юнит один.
# Собираем «синтетический город 0» — взвешенную смесь контрольных городов,
# веса подбираем по пре-периоду (lstsq + проекция на симплекс: веса >= 0, сумма 1).
# Разрыв «факт − синтетик» после вмешательства = оценка эффекта. Значимость —
# плацебо-юниты (логика TRoP): строим синтетиков для контрольных городов и
# смотрим, где в их распределении ratio post/pre MSPE лежит наш.

# %%
T4, T4_PRE, J4, SHOCK = 30, 20, 10, 0.08
f4 = 100 + 0.9 * np.arange(T4) + 7 * np.sin(0.6 * np.arange(T4))   # общий фактор рынка
levels4 = RNG.uniform(0.6, 1.5, J4)                                # масштабы городов
Y4 = np.outer(levels4, f4) + RNG.normal(0, 2.5, (J4, T4))
Y4[0, T4_PRE:] *= 1 + SHOCK                                         # хаб только в городе 0


def to_simplex(v):
    """Проекция на вероятностный симплекс (веса >= 0, сумма = 1), бисекция."""
    lo, hi = float(np.min(v)) - 1.0, float(np.max(v))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.clip(v - mid, 0, None).sum() > 1:
            lo = mid
        else:
            hi = mid
    return np.clip(v - 0.5 * (lo + hi), 0, None)


def synth_weights(j, donors):
    """Веса синтетика для юнита j по пре-периоду."""
    A = Y4[donors, :T4_PRE].T
    w_raw = np.linalg.lstsq(A, Y4[j, :T4_PRE], rcond=None)[0]
    return to_simplex(w_raw)


donors4 = list(range(1, J4))
w4 = synth_weights(0, donors4)
synth = Y4[1:].T @ w4
gap_pct = float(np.mean((Y4[0, T4_PRE:] - synth[T4_PRE:]) / synth[T4_PRE:]) * 100)
naive_pp4 = float((Y4[0, T4_PRE:].mean() / Y4[0, :T4_PRE].mean() - 1) * 100)
pre_mspe = float(np.mean((Y4[0, :T4_PRE] - synth[:T4_PRE]) ** 2))

def unit_ratio(j, donors):
    """ratio post/pre MSPE, если «лечёным» считать город j."""
    ww = synth_weights(j, donors)
    s = Y4[donors].T @ ww
    return (float(np.mean((Y4[j, T4_PRE:] - s[T4_PRE:]) ** 2))
            / max(float(np.mean((Y4[j, :T4_PRE] - s[:T4_PRE]) ** 2)), 1e-9))

ratios_pl = [unit_ratio(j, [d for d in donors4 if d != j]) for j in donors4]
ratio_tr = float(np.mean((Y4[0, T4_PRE:] - synth[T4_PRE:]) ** 2)) / pre_mspe
p_perm = float((1 + sum(r >= ratio_tr for r in ratios_pl)) / (1 + len(ratios_pl)))

print("\n4) Synthetic control (хаб в городе 0, истинный эффект = +8.0%)")
print(f"  Наивный pre-post города 0:              {naive_pp4:+6.1f}%  — съеден и рост общего фактора")
print(f"  Разрыв факт − синтетик (post, среднее): {gap_pct:+6.1f}%   (истина +8.0%)")
print(f"  Качество подгонки пре-периода: pre-MSPE = {pre_mspe:.2f} "
      f"(средний |PE| = {np.mean(np.abs(Y4[0, :T4_PRE] - synth[:T4_PRE])):.2f} при шуме 2.5)")
print("  Ненулевые веса городов: "
      + ", ".join(f"#{j + 1}: {w:.2f}" for j, w in enumerate(w4) if w > 0.01))
print(f"  Плацебо-юниты (TRoP-логика): ratio лечёного = {ratio_tr:.1f}; плацебо: "
      + ", ".join(f"{r:.2f}" for r in ratios_pl))
print(f"  Пермутационный p-value = {p_perm:.2f} (минимум при 9 плацебо = 0.10 — грубость честно признаём)")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(T4), Y4[0], color="#ee6677", lw=2, label="город 0 (хаб, факт)")
ax.plot(range(T4), synth, color="#4477aa", lw=2, label="синтетический город 0 (смесь контролей)")
ax.fill_between(range(T4_PRE, T4), Y4[0, T4_PRE:], synth[T4_PRE:], color="#ee6677", alpha=0.2,
                label=f"разрыв = эффект ≈ {gap_pct:+.1f}%")
ax.axvline(T4_PRE - 0.5, color="gray", ls=":", label="открытие хаба")
ax.set_title("Synthetic control: веса по пре-периоду, разрыв после вмешательства = эффект")
ax.set_xlabel("период"); ax.set_ylabel("метрика города"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE / "practice_6_3_synth.png", dpi=150); plt.close(fig)

# %% [markdown]
# ## Блок 5. Мини-кейс DiD+PSM: тренинг курьеров (конфаундер в назначении)
# Панель 1500 курьеров × 6 месяцев, тренинг в мае (t=5). Направление на тренинг
# зависит от X2 (прошлая активность), и X2 же задаёт индивидуальный ТРЕНД выручки
# -> безусловные параллельные тренды нарушены, условные (после контроля X) — нет.
# Истинный эффект tau = 500. Лестница: naive DiD -> DiD + ковариаты (аддитивно —
# почти не двигается; X×Post — чинит) -> матчинг по propensity (свой IRLS-logit +
# ближайший сосед с калипером 0.2 SD логита) -> DiD на отматченных.
# Грабля: в propensity попадает post-treatment ковариата — рейтинг качества
# доставок за первые 2 недели мая (тренинг на него уже повлиял).

# %%
N5, T5, G5, TAU5 = 1500, 6, 5, 500.0
Xv1 = RNG.uniform(0, 8, N5)           # стаж, годы
Xv2 = RNG.normal(0, 1, N5)            # прошлая активность: КОНФАУНДЕР
Xv3 = RNG.binomial(1, 0.3, N5)        # столица (1/0)
logit5 = -2.2 + 0.45 * Xv1 + 0.85 * Xv2 + 0.4 * Xv3
D5 = (RNG.uniform(size=N5) < 1 / (1 + np.exp(-logit5))).astype(float)
slope5 = 12.0 * Xv2                   # тренд выручки зависит от конфаундера
Y5 = np.empty((N5, T5))
for t in range(1, T5 + 1):
    Y5[:, t - 1] = (800 + 25 * Xv1 + 80 * Xv2 + 40 * Xv3 + 18 * t
                    + slope5 * t + RNG.normal(0, 60, N5)
                    + TAU5 * D5 * (t >= G5))

pre5, post5 = slice(0, G5 - 1), slice(G5 - 1, T5)
dy5 = Y5[:, post5].mean(axis=1) - Y5[:, pre5].mean(axis=1)


def did_2x2(mask=None, w=None):
    """DiD на (возможно взвешенной) подвыборке: dY теста − dY контроля."""
    if mask is None:
        mask = np.ones(N5, bool)
    if w is None:
        w = np.ones(N5)
    wt, wc = w * D5 * mask, w * (1 - D5) * mask
    est = np.average(dy5, weights=wt) - np.average(dy5, weights=wc)
    se_ = np.sqrt(dy5[D5 * mask > 0].var(ddof=1) / wt.sum()
                  + dy5[(1 - D5) * mask > 0].var(ddof=1) / wc.sum())
    return float(est), float(se_)


est_naive, se_naive = did_2x2()

# DiD-регрессии: (а) ковариаты аддитивно, (б) ковариаты в тренд (X*Post)
li = np.repeat(np.arange(N5), T5)
lt = np.tile(np.arange(1, T5 + 1), N5)
lpost = (lt >= G5).astype(float)
lD = D5[li]
Xa = np.column_stack([np.ones(N5 * T5), lpost, lD, lpost * lD,
                      Xv1[li], Xv2[li], Xv3[li]])
ba, sa, _, pa = ols(Y5.ravel(), Xa)
Xb = np.column_stack([np.ones(N5 * T5), lpost, lD, lpost * lD,
                      (Xv1 * lpost)[li], (Xv2 * lpost)[li], (Xv3 * lpost)[li]])
bb, sb, _, pb = ols(Y5.ravel(), Xb)


def logit_irls(Xm, y, iters=50):
    """Логистическая регрессия (propensity) через IRLS + lstsq."""
    b = np.zeros(Xm.shape[1])
    for _ in range(iters):
        eta = np.clip(Xm @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = np.sqrt(np.clip(p * (1 - p), 1e-9, None))
        z = eta + (y - p) / np.clip(p * (1 - p), 1e-9, None)
        b_new = np.linalg.lstsq(Xm * w[:, None], z * w, rcond=None)[0]
        if np.max(np.abs(b_new - b)) < 1e-10:
            return b_new
        b = b_new
    return b


def match_nn(ps_t, ps_c, caliper):
    """Ближайший сосед по логиту PS (с заменой) + калипер -> пары и веса контроля."""
    used = {}
    for i in np.argsort(ps_t)[::-1]:           # сначала самые «лечёные»
        d = np.abs(ps_c - ps_t[i])
        j = int(np.argmin(d))
        if d[j] <= caliper:
            used[int(i)] = j
    w_ctrl = np.zeros(len(ps_c))
    for j in used.values():
        w_ctrl[j] += 1.0
    return used, w_ctrl


def smd(v, w):
    """Стандартизированная разность средних (с весами)."""
    mt = np.average(v, weights=w * D5)
    mc = np.average(v, weights=w * (1 - D5))
    vt = np.average((v - mt) ** 2, weights=w * D5)
    vc = np.average((v - mc) ** 2, weights=w * (1 - D5))
    return (mt - mc) / np.sqrt((vt + vc) / 2)


Xm = np.column_stack([np.ones(N5), Xv1, Xv2, Xv3])
ps5 = 1 / (1 + np.exp(-(Xm @ logit_irls(Xm, D5))))
lps5 = np.log(ps5 / (1 - ps5))
used5, w_ctrl5 = match_nn(lps5[D5 == 1], lps5[D5 == 0], 0.2 * np.std(lps5))
matched5 = np.zeros(N5, bool)
matched5[list(used5.keys())] = True
w5 = D5 * matched5 + w_ctrl5 * (1 - D5)         # лечёные весом 1, контроли — кратность
est_match, _ = did_2x2(matched5, w5)
ess5 = w_ctrl5.sum() ** 2 / np.sum(w_ctrl5 ** 2)

# бутстреп CI для DiD на отматченных (ресемплим пары)
pairs5 = np.array([[i, j] for i, j in used5.items()], int)
boot5 = []
for _ in range(1000):
    sel = RNG.integers(0, len(pairs5), len(pairs5))
    boot5.append(dy5[pairs5[sel, 0]].mean() - dy5[pairs5[sel, 1]].mean())
ci_lo5, ci_hi5 = np.percentile(boot5, [2.5, 97.5])

# ГРАБЛЯ: post-treatment ковариата в propensity — рейтинг качества первых
# 2 недель мая: коррелирует с выручкой мая, на которую тренинг УЖЕ повлиял
W_post = 0.6 * (Y5[:, G5 - 1] - Y5[:, G5 - 1].mean()) / Y5[:, G5 - 1].std() \
    + 0.8 * RNG.normal(size=N5)
Xm_bad = np.column_stack([np.ones(N5), Xv1, Xv2, Xv3, W_post])
ps_bad = 1 / (1 + np.exp(-(Xm_bad @ logit_irls(Xm_bad, D5))))
lps_bad = np.log(ps_bad / (1 - ps_bad))
used_bad, w_bad = match_nn(lps_bad[D5 == 1], lps_bad[D == 0 if False else 0], 0)  # placeholder
used_bad, w_bad = match_nn(lps_bad[D5 == 1], lps_bad[D5 == 0], 0.2 * np.std(lps_bad))
matched_bad = np.zeros(N5, bool)
matched_bad[list(used_bad.keys())] = True
w5_bad = D5 * matched_bad + w_bad * (1 - D5)
est_bad, _ = did_2x2(matched_bad, w5_bad)

smd_before = [smd(Xv1, np.ones(N5)), smd(Xv2, np.ones(N5)), smd(Xv3, np.ones(N5))]
smd_after = [smd(Xv1, w5), smd(Xv2, w5), smd(Xv3, w5)]

ladder = pd.DataFrame({
    "оценщик": ["Naive DiD (2×2)", "DiD + ковариаты (аддитивно)",
                "DiD + X×Post (ковариаты в тренд)", "PSM (NN + калипер) + DiD",
                "PSM c post-treatment ковариатой"],
    "оценка": [est_naive, ba[3], bb[3], est_match, est_bad],
    "% истины (tau=500)": [100 * v / TAU5 for v in [est_naive, ba[3], bb[3], est_match, est_bad]],
}).round(1)

print("\n5) DiD+PSM: тренинг курьеров (конфаундер X2 в назначении, истинный tau = 500)")
print(f"  Лечёных {D5.sum():.0f} из {N5} ({D5.mean():.0%}); SMD до матчинга (стаж, активность, столица): "
      + ", ".join(f"{s:+.2f}" for s in smd_before))
print(f"  Naive DiD:                    {est_naive:7.1f} (SE={se_naive:.1f}) -> "
      f"{100 * (est_naive - TAU5) / TAU5:+.0f}% от истины: безусловные тренды непараллельны")
print(f"  Ковариаты аддитивно:          {ba[3]:7.1f} (p={pa[3]:.2g}) — почти не сдвинулось:")
print("    time-invariant X не различает «состав групп» и «тренд», DiD-коэффициент его не видит")
print(f"  Ковариаты в тренд (X×Post):   {bb[3]:7.1f} (p={pb[3]:.2g}) — тренды выровнены, оценка у истины")
print(f"  Матчинг: пар {len(used5)} из {int(D5.sum())} лечёных (потеря "
      f"{100 * (1 - len(used5) / D5.sum()):.0f}% на калипере/common support); "
      f"ESS контроля = {ess5:.0f} из {int(w_ctrl5.sum())}")
print(f"  SMD после матчинга: " + ", ".join(f"{s:+.2f}" for s in smd_after) + " (порог 0.1)")
print(f"  PSM + DiD:                    {est_match:7.1f} (бутстреп 95% CI {ci_lo5:.0f}..{ci_hi5:.0f})")
print(f"  ГРАБЛЯ (post-treatment в PS): {est_bad:7.1f} -> {100 * (est_bad - TAU5) / TAU5:+.0f}% от истины:")
print("    матчинг по переменной, на которую лечение уже повлияло, подтягивает контролей")
print("    «под лечёных» и систематически вычитает часть эффекта")
print("\n  Лестница оценщиков (истина 500):")
print(ladder.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
ax.barh(["стаж", "активность", "столица"], smd_before, color="#cc3311", label="до матчинга")
ax.barh(["стаж", "активность", "столица"], smd_after, color="#228833", label="после")
ax.axvline(0.1, color="gray", ls="--", lw=1); ax.axvline(-0.1, color="gray", ls="--", lw=1)
ax.set_title("Love plot: SMD до/после матчинга (порог ±0.1)")
ax.set_xlabel("SMD"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
ax = axes[1]
ax.barh(ladder["оценщик"], ladder["оценка"],
        color=["#cc3311", "#ee7733", "#228833", "#228833", "#cc3311"])
ax.axvline(TAU5, color="k", ls="--", label="истина tau = 500")
for i, (v, pct) in enumerate(zip(ladder["оценка"], ladder["% истины (tau=500)"])):
    ax.text(v + 8, i, f"{v:.0f} ({pct:+.0f}%)", va="center", fontsize=8)
ax.set_title("Лестница DiD+PSM: кто ближе всех к истине")
ax.set_xlabel("оценка эффекта, ₽"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
fig.tight_layout(); fig.savefig(HERE / "practice_6_3_psm.png", dpi=150); plt.close(fig)

# %% [markdown]
# ## Итог
# - DiD работает, пока тренды параллельны: тренд и сезонность уходят контролю,
#   эффект — коэффициенту при Post×Treat. Наивный pre-post смешивает их.
# - Pretrend-тест (event study, плацебо-периоды) обязателен, но это необходимое
#   условие, а не доказательство параллельности после лечения.
# - Staggered + неоднородные эффекты ломают наивный TWFE: сравнивайте когорты
#   с ещё-не-лечёными/чистым контролем и агрегируйте ATT(g,t) — логика CS(2021).
# - Один юнит — синтетический контроль: смесь контролей по пре-периоду, разрыв
#   после вмешательства; значимость — плацебо-юнитами (TRoP-логика).
# - Комбинация PSM+DiD покупает условные параллельные тренды ценой +SE и потери
#   выборки на common support; матчиться ТОЛЬКО по pre-treatment ковариатам,
#   pretrends проверять на отматченных. Современный стандарт — doubly robust DiD.

print("\nГотово. PNG: practice_6_3_did / event_study / staggered / synth / psm — рядом со скриптом.")
