# -*- coding: utf-8 -*-
"""Практика 6.4 — Наблюдательные данные: PSM, IPW, doubly robust, IV
(сквозной кейс «ЕдаДома»: подписка «ЕдаПлюс» — бесплатная доставка).

Не верь — проверяй симуляцией: истинный эффект зашит в генератор.

Что делаем:
1) СИНТЕТИКА С ИЗВЕСТНОЙ ИСТИНОЙ: конфаундеры (активность в предпериоде,
   доход района, мегаполис) толкают и в подписку, и в заказы.
   Истина: эффект подписки = +1.0 заказ/мес (однородный, чтобы ATE=ATT=LATE).
2) ЛЕСТНИЦА К ИСТИНЕ: наивная разность -> регрессия с ковариатами ->
   PSM (propensity руками: логистическая регрессия через numpy-Ньютон;
   матчинг 1:1 по логиту PS с caliper = 0.2*SD) -> IPW (псевдопопуляция,
   экстремальные веса, ESS, транкация) -> doubly robust (AIPW).
   Диагностика PSM: Love plot (SMD до/после), common support, ESS,
   кого выбросил калипер (смена эстиманда ATT -> SATT).
   Двойная устойчивость: 4 сценария «какая из моделей врёт».
3) ГРАБЛЯ: матчинг по post-treatment ковариате (медиатору) убивает эффект.
4) IV / ENCOURAGEMENT DESIGN: рандомный пуш Z подталкивает к подписке D
   (влияет частично — «мягкий рандомизатор»). Наивная регрессия смещена,
   Wald = истинному LATE; 2SLS руками совпадает с Wald (один инструмент).
   Слабый инструмент (Z почти не двигает D): первый этап F<10, знаменатель
   Wald ~ 0, оценка взрывается — 300 симуляций сильного против слабого IV.

Запуск:  python3 practice_6_4.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# # Практика 6.4 — лестница от наивного сравнения к doubly robust
# Всё на numpy/scipy: propensity-модель пишем руками, чтобы она не была
# «чёрным ящиком из sklearn», — логистическая регрессия это 10 строк.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})

BLUE, RED, GREEN, GREY, ORANGE = "#4C72B0", "#C44E52", "#55A868", "#8c8c8c", "#DD8452"


# %% [markdown]
# ## 0. Данные «ЕдаДома»: подписка «ЕдаПлюс» и заказы
# Подписку никто не рандомизировал: её берут активные пользователи из богатых
# районов мегаполисов — и они же заказывают больше БЕЗ всякой подписки.

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_data(n=4000, seed=6401):
    """Возвращает X (3 конфаундера), D (подписка), Y (заказов/мес после).

    DGP:
      activity  ~ N(4, 2)          -- заказов/мес в предпериоде
      income    ~ N(0, 1)          -- доход района (стандартизованный)
      mega      ~ Bernoulli(0.3)   -- мегаполис
      e(D=1|X)  = sigmoid(-2.2 + 0.45*activity + 0.8*income + 0.7*mega)
      Y         = 3 + 0.6*activity + 0.9*income + 0.7*mega + 1.0*D + N(0, 1)
    Истинный эффект = +1.0 (однородный). Наивное сравнение обязано быть
    смещено ВВЕРХ: activity и income повышают и P(D=1), и Y.
    """
    rng = np.random.default_rng(seed)
    activity = np.clip(rng.normal(4, 2, n), 0, None)
    income = rng.normal(0, 1, n)
    mega = rng.binomial(1, 0.3, n)
    X = np.column_stack([activity, income, mega])
    e_true = sigmoid(-2.2 + 0.45 * activity + 0.8 * income + 0.7 * mega)
    D = rng.binomial(1, e_true)
    Y = (3 + 0.6 * activity + 0.9 * income + 0.7 * mega
         + 1.0 * D + rng.normal(0, 1, n))
    return rng, X, D.astype(float), Y, e_true, ["activity", "income", "mega"]


def smd(x, d, w=None):
    """Стандартизованная разность средних (treated=1 vs control=0).

    w -- веса контроля (например, счётчик повторного использования пары).
    Возвращает (mean_t - mean_c) / sqrt((var_t + var_c) / 2).
    """
    t = d == 1
    c = d == 0
    mt, vt = x[t].mean(), x[t].var()
    if w is None:
        mc, vc = x[c].mean(), x[c].var()
    else:
        mc = np.average(x[c], weights=w)
        vc = np.average((x[c] - mc) ** 2, weights=w)
    pooled = np.sqrt((vt + vc) / 2.0)
    return (mt - mc) / pooled if pooled > 0 else 0.0


def fit_logistic(X, y, n_iter=60, tol=1e-11):
    """Логистическая регрессия «руками»: метод Ньютона (градиент + гессиан).

    Возвращает (beta, X_design): X уже стандартизована, добавлен интерсепт.
    """
    mu, sd = X.mean(0), X.std(0)
    Xs = (X - mu) / sd
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    beta = np.zeros(Xd.shape[1])
    for _ in range(n_iter):
        p = sigmoid(Xd @ beta)
        w = p * (1 - p)
        grad = Xd.T @ (y - p)
        hess = (Xd * w[:, None]).T @ Xd
        step = np.linalg.solve(hess + 1e-10 * np.eye(len(beta)), grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta, Xd


def ols(X, y):
    """МНК без sklearn: возвращает коэффициенты [intercept, ...]."""
    Xd = np.column_stack([np.ones(len(X)), X]) if X.ndim > 1 else \
        np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(Xd, y, rcond=None)[0]


def match_pairs(logit_ps, D, caliper):
    """Жадный 1:1 матчинг с заимствованием контроля по |logit-разности|.

    Возвращает (idx_t, idx_c) пар, прошедших caliper, и маску оставленных
    treated (для подсчёта выброшенных).
    """
    it, = np.where(D == 1)
    ic, = np.where(D == 0)
    dist = np.abs(logit_ps[it][:, None] - logit_ps[ic][None, :])
    j = dist.argmin(axis=1)
    dmin = dist[np.arange(len(it)), j]
    keep = dmin <= caliper
    return it[keep], ic[j[keep]], keep


def ess(w):
    """Effective Sample Size: (sum w)^2 / sum w^2."""
    return w.sum() ** 2 / (w ** 2).sum()


# %% [markdown]
# ## 1. Лестница к истине: пять оценок одного эффекта

rng, X, D, Y, e_true, NAMES = make_data()
n = len(Y)
truth = 1.0
print("=" * 88)
print("1) ЛЕСТНИЦА К ИСТИНЕ. DGP: e(D=1|X)=sigmoid(-2.2+0.45*act+0.8*inc+0.7*mega),")
print("   Y = 3 + 0.6*act + 0.9*inc + 0.7*mega + 1.0*D + шум. ИСТИННЫЙ ЭФФЕКТ = +1.0")
print("=" * 88)

# --- ступень 0: наивная разность
naive = Y[D == 1].mean() - Y[D == 0].mean()

# --- ступень 1: регрессия с ковариатами (правильная спецификация)
b_ols = ols(X, Y)
reg_covariate = b_ols[-1] if False else b_ols[1 + NAMES.index("activity")] * 0  # placeholder
reg_covariate = ols(X, Y)[-1]  # не используется; эффект считаем честно ниже
# эффект из регрессии Y ~ X + D:
b_full = ols(np.column_stack([X, D]), Y)
reg_adjusted = b_full[-1]

# --- ступень 2: PSM (propensity руками -> матчинг по логиту с caliper)
beta_ps, Xd = fit_logistic(X, D)
e_hat = sigmoid(Xd @ beta_ps)
logit = np.log(e_hat / (1 - e_hat))
caliper = 0.2 * logit.std()
idx_t, idx_c, kept = match_pairs(logit, D, caliper)
psm_att = (Y[idx_t] - Y[idx_c]).mean()
dropped = (~kept).sum()
share_dropped = dropped / (D == 1).sum()
w_ctrl = np.bincount(idx_c, minlength=n).astype(float)
ess_matched = ess(w_ctrl[w_ctrl > 0])

smd_before = [smd(X[:, k], D) for k in range(X.shape[1])]
smd_after = [smd(X[:, k], D, w=np.bincount(idx_c, minlength=n)) for k in range(X.shape[1])]

# --- ступень 3: IPW (псевдопопуляция) + транкация экстремальных весов
w_t, w_c = D / e_hat, (1 - D) / (1 - e_hat)
ipw = (np.sum(D * Y / e_hat) / np.sum(D / e_hat)
       - np.sum((1 - D) * Y / (1 - e_hat)) / np.sum((1 - D) / (1 - e_hat)))
w_t_tr, w_c_tr = w_t.copy(), w_c.copy()
q_t, q_c = np.quantile(w_t[D == 1], 0.99), np.quantile(w_c[D == 0], 0.99)
w_t_tr[D == 1] = np.minimum(w_t[D == 1], q_t)
w_c_tr[D == 0] = np.minimum(w_c[D == 0], q_c)
ipw_tr = (np.sum(w_t_tr * Y) / np.sum(w_t_tr)
          - np.sum(w_c_tr * Y) / np.sum(w_c_tr))

# --- ступень 4: doubly robust (AIPW)
def outcome_models(cols):
    """Модели исхода mu0/mu1 по выбранным колонкам (для «поломок»)."""
    Xs = X[:, cols]
    mu0 = ols(Xs[D == 0], Y[D == 0])
    mu1 = ols(Xs[D == 1], Y[D == 1])
    m0 = np.column_stack([np.ones(sum(D == 0)), Xs[D == 0]]) @ mu0
    m1 = np.column_stack([np.ones(sum(D == 1)), Xs[D == 1]]) @ mu1
    m0f = np.column_stack([np.ones(n), Xs]) @ mu0   # предсказание всем
    m1f = np.column_stack([np.ones(n), Xs]) @ mu1
    return m0f, m1f


def aipw(m0, m1, e):
    psi = m1 - m0 + D * (Y - m1) / e - (1 - D) * (Y - m0) / (1 - e)
    return psi.mean()


all_cols = list(range(X.shape[1]))
bad_cols = [NAMES.index("income"), NAMES.index("mega")]  # выкинули activity
m0_full, m1_full = outcome_models(all_cols)
m0_bad, m1_bad = outcome_models(bad_cols)
_, Xd_bad = fit_logistic(X[:, bad_cols], D)
e_bad = sigmoid(Xd_bad @ fit_logistic(X[:, bad_cols], D)[0])
dr_full = aipw(m0_full, m1_full, e_hat)
dr_ps_bad = aipw(m0_full, m1_full, e_bad)       # propensity врёт, исход верен
dr_om_bad = aipw(m0_bad, m1_bad, e_hat)         # propensity верна, исход врёт
dr_both_bad = aipw(m0_bad, m1_bad, e_bad)       # врут обе

print(f"   подписались: {int(D.sum())} из {n} ({D.mean():.0%}), "
      f"медиана e_hat = {np.median(e_hat):.2f}")
print()
print(f"   {'оценщик':38s} {'эффект':>8s} {'ошибка':>9s}")
print(f"   {'-'*38} {'-'*8} {'-'*9}")
rows = [("наивная разность (D=1 vs D=0)", naive),
        ("регрессия Y ~ X + D", reg_adjusted),
        (f"PSM: 1:1 по логиту, caliper {caliper:.3f}", psm_att),
        ("IPW (Хájek, все веса)", ipw),
        ("IPW после транкации p99", ipw_tr),
        ("doubly robust (AIPW)", dr_full)]
for name, val in rows:
    print(f"   {name:38s} {val:+8.3f} {val - truth:+9.3f}")
print()
print(f"   ИСТИНА: +1.000. Наивная разность завышена на {(naive - truth):.2f} --")
print("   это и есть selection bias: подписались и так самые активные.")

# %% [markdown]
# ## 2. Диагностика PSM: баланс, overlap, ESS, кого выбросили

print("=" * 88)
print("2) ЧЕСТЬ МЕТОДА: диагностика матчинга (без неё PSM -- ритуал, не метод)")
print("=" * 88)
print(f"   SMD (|.|) по ковариатам, порог 0.1 (Love plot ниже):")
for k, name in enumerate(NAMES):
    flag_b = "OK  " if abs(smd_before[k]) < 0.1 else "ПЛОХО"
    flag_a = "OK  " if abs(smd_after[k]) < 0.1 else "ПЛОХО"
    print(f"     {name:10s} до: {abs(smd_before[k]):5.2f} [{flag_b}]  "
          f"после: {abs(smd_after[k]):5.2f} [{flag_a}]")
print(f"   калипер {caliper:.3f} (0.2*SD логита PS) выбросил {dropped} treated "
      f"({share_dropped:.0%}): для них нет похожего контроля,")
print(f"   вывод -- не про всех подписчиков: эстиманд сместился ATT -> SATT (по сегменту).")
print(f"   ESS контроля после матчинга с заимствованием: {ess_matched:.0f} "
      f"(строк контроля: {int((D == 0).sum())}, пар: {len(idx_t)})")
print(f"   IPW, веса: max treated 1/e = {w_t[D == 1].max():.1f}, "
      f"max control 1/(1-e) = {w_c[D == 0].max():.1f}")
print(f"   ESS IPW: treated {(ess(w_t[D == 1])):.0f}/{int((D == 1).sum())}, "
      f"control {ess(w_c[D == 0]):.0f}/{int((D == 0).sum())} --")
print("   экстремальные веса съедают выборку: несколько юзеров решают всё.")
print(f"   после транкации p99: ESS control {ess(w_c_tr[D == 0]):.0f}, "
      f"оценка {ipw_tr:+.3f} (чуть смещение -- плата за устойчивость).")

print()
print("   DOUBLE ROBUST: достаточно ОДНОЙ верной модели")
print(f"     обе верны:                     {dr_full:+.3f}")
print(f"     propensity врёт, исход верен:  {dr_ps_bad:+.3f}  <- держится")
print(f"     propensity верна, исход врёт:  {dr_om_bad:+.3f}  <- держится")
print(f"     врут обе:                      {dr_both_bad:+.3f}  <- уехала")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
# Love plot
ax = axes[0]
yy = np.arange(len(NAMES))
ax.scatter(smd_before, yy, s=70, color=RED, zorder=3, label="до матчинга")
ax.scatter(smd_after, yy, s=70, color=GREEN, zorder=3, label="после матчинга")
for k in range(len(NAMES)):
    ax.plot([smd_before[k], smd_after[k]], [k, k], color=GREY, lw=1.2, zorder=2)
ax.axvline(0.1, color="k", ls="--", lw=1)
ax.axvline(-0.1, color="k", ls="--", lw=1)
ax.set_yticks(yy, NAMES)
ax.set_xlabel("SMD (стандартизованная разность средних)")
ax.set_title("Love plot: матчинг прижал ковариаты в трубу |SMD| < 0.1")
ax.legend(loc="lower right", fontsize=9)
# common support
ax = axes[1]
bins = np.linspace(logit.min(), logit.max(), 40)
ax.hist(logit[D == 0], bins=bins, density=True, alpha=0.65, color=BLUE, label="контроль")
ax.hist(logit[D == 1], bins=bins, density=True, alpha=0.65, color=RED, label="treated")
ax.set_xlabel("logit(propensity score)")
ax.set_title("Common support: хвосты без пересечения -- зоны,\nгде контрфакта не существует (их съест caliper)")
ax.legend(fontsize=9)
# IPW weights
ax = axes[2]
ax.hist(w_c[D == 0], bins=60, alpha=0.65, color=BLUE, label=f"контроль: 1/(1-e), max={w_c[D==0].max():.1f}")
ax.hist(w_t[D == 1], bins=60, alpha=0.65, color=RED, label=f"treated: 1/e, max={w_t[D==1].max():.1f}")
ax.axvline(q_c, color=BLUE, ls="--", lw=1.2, label="транкация p99 (контроль)")
ax.set_xlabel("вес IPW")
ax.set_title("Экстремальные веса IPW: горстка юзеров с весом 20+ решает всё;\nESS падает, дисперсия взлетает")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "practice_6_4_love_plot.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 3. Грабля: матчинг по post-treatment ковариате

print("=" * 88)
print("3) ГРАБЛЯ: матчинг по МЕДИАТОРУ (ковариата ПОСЛЕ подписки)")
print("=" * 88)
# Новый мир: эффект подписки работает ЧЕРЕЗ медиатор M (заказы «здоровой еды»),
# M = 1.2*D + 0.5*activity + шум; Y = 3 + 0.6*activity + 0.9*income + 0.7*mega
#       + 0.6*M + шум. Полный эффект D на Y = 0.6*1.2 = +0.72.
rng2, X, D, Y, e_true, NAMES = make_data(seed=6402)
M = 1.2 * D + 0.5 * X[:, 0] + rng2.normal(0, 0.5, n)
Y = 3 + 0.6 * X[:, 0] + 0.9 * X[:, 1] + 0.7 * X[:, 2] + 0.6 * M + rng2.normal(0, 1, n)
total_effect_true = 0.6 * 1.2

beta_ps, Xd = fit_logistic(X, D)
e_hat = sigmoid(Xd @ beta_ps)
logit = np.log(e_hat / (1 - e_hat))
caliper = 0.2 * logit.std()

# правильно: PS только по конфаундерам «до»
it, ic, _ = match_pairs(logit, D, caliper)
att_ok = (Y[it] - Y[ic]).mean()

# грабля: добавляем в PS медиатор M («у него же видно, что он подписчик!»)
X_med = np.column_stack([X, M])
beta_m, Xd_m = fit_logistic(X_med, D)
e_m = sigmoid(Xd_m @ beta_m)
logit_m = np.log(e_m / (1 - e_m))
cal_m = 0.2 * logit_m.std()
it2, ic2, _ = match_pairs(logit_m, D, cal_m)
att_med = (Y[it2] - Y[ic2]).mean()

print(f"   ИСТИНА (полный эффект через медиатор): {total_effect_true:+.3f}")
print(f"   PS по конфаундерам «до»:              {att_ok:+.3f}  <- ок")
print(f"   PS «до» + медиатор M (post-treatment): {att_med:+.3f}  <- эффект «исчез»")
print("   Механика: матчинг по M подбирает контролю «такого же по M»,")
print("   т.е. фиксирует путь D -> M -> Y. Мы заблокировали медиатор и")
print("   спросили «а сколько даёт подписка, если её главный канал выключить?».")
print("   Контроль post-treatment ковариат = вопрос не о том эффекте.")

# %% [markdown]
# ## 4. IV: encouragement design «рандомный пуш», Wald и 2SLS

print("=" * 88)
print("4) IV / ENCOURAGEMENT: пуш Z -> подписка D -> заказы Y (конфаундер U)")
print("=" * 88)


def make_iv(n, beta_z, seed):
    """U влияет и на D, и на Y (ненаблюдаемый). Z рандомен: Bernoulli(0.5)."""
    r = np.random.default_rng(seed)
    U = r.normal(0, 1, n)
    Z = r.binomial(1, 0.5, n)
    D = r.binomial(1, sigmoid(-1.0 + beta_z * Z + 1.0 * U)).astype(float)
    Y = 2 + 1.0 * D + 1.0 * U + 0.5 * r.normal(0, 1, n)
    return Z.astype(float), D, Y


def wald(Z, D, Y):
    return ((Y[Z == 1].mean() - Y[Z == 0].mean())
            / (D[Z == 1].mean() - D[Z == 0].mean()))


def first_stage_F(Z, D):
    """F-статистика коэффициента при Z в регрессии D ~ Z (F = t^2)."""
    zbar, dbar = Z.mean(), D.mean()
    b = ((Z - zbar) * (D - dbar)).sum() / ((Z - zbar) ** 2).sum()
    resid = D - (dbar + b * (Z - zbar))
    sigma2 = (resid ** 2).sum() / (len(D) - 2)
    se = np.sqrt(sigma2 / ((Z - zbar) ** 2).sum())
    return (b / se) ** 2, b


Z, D, Y = make_iv(4000, beta_z=1.1, seed=6410)
f_stat, first_stage_b = first_stage_F(Z, D)
w_strong = wald(Z, D, Y)
naive_iv = ols(Z * 0 + D, Y)[-1]           # OLS Y ~ D (наивный)
b2 = ols(Z, D)                              # 1-я стадия
d_hat = b2[0] + b2[1] * Z                   # предсказанный D
twosls = ols(d_hat, Y)[1]                   # 2-я стадия: Y ~ D_hat
print(f"   СИЛЬНЫЙ инструмент (beta_z=1.1), n=4000:")
print(f"     первый этап: E[D|Z=1]-E[D|Z=0] = "
      f"{D[Z == 1].mean() - D[Z == 0].mean():+.3f}, F = {f_stat:.0f} (>10: ок)")
print(f"     наивная регрессия Y~D:  {naive_iv:+.3f}  <- смещена U (вовлечённость)")
print(f"     Wald:                    {w_strong:+.3f}")
print(f"     2SLS (две регрессии):   {twosls:+.3f}  == Wald: один инструмент --")
print("     точечные оценки совпадают, 2SLS ещё даёт SE и работает с ковариатами")
print(f"     ИСТИНА: +1.000 (compliance неполная -- Wald оценивает LATE)")

Zw, Dw, Yw = make_iv(4000, beta_z=0.22, seed=6411)
f_weak, b_weak = first_stage_F(Zw, Dw)
w_weak = wald(Zw, Dw, Yw)
print()
print(f"   СЛАБЫЙ инструмент (beta_z=0.22):")
print(f"     первый этап: сдвиг = {Dw[Zw == 1].mean() - Dw[Zw == 0].mean():+.4f}, "
      f"F = {f_weak:.1f} (<10: инструмент слаб)")
print(f"     Wald этой реализации: {w_weak:+.2f} (знаменатель почти ноль)")

# 300 симуляций: распределение Wald для сильного и слабого инструментов
sim_strong, sim_weak = [], []
for s in range(300):
    z, d, y = make_iv(1000, 1.1, seed=7000 + s)
    sim_strong.append(wald(z, d, y))
    z, d, y = make_iv(1000, 0.22, seed=8000 + s)
    sim_weak.append(wald(z, d, y))
sim_strong, sim_weak = np.array(sim_strong), np.array(sim_weak)

print()
print(f"   300 симуляций (n=1000):                   сильный IV   слабый IV")
print(f"     медиана Wald:                           {np.median(sim_strong):+9.3f}  "
      f"{np.median(sim_weak):+9.3f}")
q1s, q3s = np.quantile(sim_strong, [0.25, 0.75])
q1w, q3w = np.quantile(sim_weak, [0.25, 0.75])
print(f"     межквартильная ширина (Q3-Q1):          {q3s - q1s:9.3f}  {q3w - q1w:9.3f}")
print(f"     доля |Wald| > 3 (безумные оценки):      "
      f"{np.mean(np.abs(sim_strong) > 3):9.1%}  {np.mean(np.abs(sim_weak) > 3):9.1%}")
print("     слабый инструмент делит шум числителя на почти-ноль в знаменателе:")
print("     амплитуда ошибки растёт на порядки. F>10 -- фильтр, а не формальность.")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax = axes[0]
ax.hist(np.clip(sim_strong, -2, 4), bins=40, color=GREEN, alpha=0.8,
        label=f"сильный IV (F~{f_stat:.0f} на n=4000)")
ax.hist(np.clip(sim_weak, -2, 4), bins=40, color=RED, alpha=0.6,
        label="слабый IV (F<10), хвосты обрезаны на [-2; 4]")
ax.axvline(1.0, color="k", ls="--", lw=1.4, label="истина +1.0")
ax.set_xlabel("оценка Wald")
ax.set_title("Распределение Wald-оценки, 300 симуляций: слабый инструмент --\nэто деление шума на почти-ноль")
ax.legend(fontsize=9)
ax = axes[1]
bins = np.linspace(-0.2, 1.2, 30)
ax.hist(D[Z == 0], bins=bins, alpha=0.65, color=BLUE, label="Z=0 (без пуша)")
ax.hist(D[Z == 1], bins=bins, alpha=0.65, color=RED, label="Z=1 (с пушем)")
ax.set_xlabel("D (подписка)")
ax.set_xticks([0, 1])
ax.set_title(f"Первый этап, сильный IV: пуш реально двигает подписку "
             f"(+{D[Z == 1].mean() - D[Z == 0].mean():.2f}, F={f_stat:.0f})")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "practice_6_4_iv.png", dpi=150)
plt.close(fig)

print()
print("Графики: practice_6_4_love_plot.png (Love plot, overlap, веса IPW),")
print("         practice_6_4_iv.png (Взрыв Wald на слабом инструменте)")
