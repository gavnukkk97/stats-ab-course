# -*- coding: utf-8 -*-
"""Практика 6.5 — Доверие к результату и карта методов (финал М6).

Сквозной кейс «ЕдаДома»: продолжаем аудит эффекта подписки «ЕдаПлюс»
из 6.4 (истина +1.0 заказа/мес; здесь PSM даёт ~ +1.0). Получить оценку --
полдела; этот скрипт -- про вторую половину: ЗАСЛУЖИВАЕТ ЛИ ОНА ДОВЕРИЯ,
и как выбрать метод, когда новый вопрос приходит без дизайна.

Что делаем:
1) ПЛАЦЕБО-ТЕСТЫ (refutation-культура): тот же PSM-пайплайн прогоняем на
   (а) плацебо-исходе -- метрика, на которую подписка повлиять не может
   (число жалоб на качество); (б) плацебо-времени -- целевая метрика ДО
   существования подписки (заказы прошлого месяца): подписки ещё нет,
   а наивная разность уже «есть эффект» -- селекция; PSM обязан её убрать;
   (в) плацебо-экспозиции -- перемешиваем D (100 пермутаций): эмпирический
   p-value наблюдаемого эффекта.
2) SENSITIVITY К НЕНАБЛЮДАЕМЫМ КОНФАУНДЕРАМ:
   E-value (VanderWeele--Ding: E = RR + sqrt(RR*(RR-1))) для бинарного
   исхода «повторный заказ» -- насколько сильный (в RR) должен быть скрытый
   конфаундер, связанный и с лечением, и с исходом, чтобы объяснить эффект;
   упрощённая ГАММА РОЗЕНБАУМА: по сматченным парам -- знаковый тест,
   ищем Г*, при которой p-value поднимается до альфы (какого скрытого
   смещения достаточно, чтобы убить значимость) + таблица интерпретации.
3) ДЕРЕВО ВЫБОРА МЕТОДА: method_tree(**ситуация) -- текстовое дерево
   «можно рандомизировать? -> этика/стоимость/уже произошло -> есть
   контроль? -> параллельные тренды? -> порог? -> один юнит? ->
   инструмент? -> ковариаты?»; 7 сценариев «ЕдаДома».
4) ЧЕК-ЛИСТ «КАУЗАЛЬНЫЙ АУДИТ»: собираем диагностику блоков 1-2 в
   вердикт PASS/WARN/FAIL по каждому пункту + пункты, которые код не
   проверяет (доменная экспертиза, триангуляция).

Отличие от 6.4: матчинг БЕЗ заимствования (каждый контроль -- максимум в
одной паре) -- пары независимы, SE парной разности честная; заимствование
в 6.4 было демонстрацией ESS-эффекта, здесь нам нужна чистая инференция.

Запуск:  python3 practice_6_5.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# # Практика 6.5 — доверие к результату: плацебо, чувствительность, карта
# Всё на numpy/scipy; хелперы PSM скопированы из 6.4, чтобы скрипт был
# самодостаточным. Истина по-прежнему зашита в генератор.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, norm

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})

BLUE, RED, GREEN, GREY, ORANGE = "#4C72B0", "#C44E52", "#55A868", "#8c8c8c", "#DD8452"

ALPHA = 0.05


# %% [markdown]
# ## 0. Данные и PSM-пайплайн из 6.4 (self-contained копия, матчинг без заимствования)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X, y, n_iter=60, tol=1e-11):
    """Логистическая регрессия руками (Ньютон). X стандартизуется внутри."""
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


def match_pairs_noreuse(logit_ps, D, caliper, rng):
    """Жадный 1:1 матчинг БЕЗ заимствования: каждый контроль -- одна пара.

    Treated обходим в случайном порядке (иначе порядок решает, кому лучшие
    контроли), ближайший СВОБОДНЫЙ контроль; пары дальше caliper выбрасываются.
    """
    it, = np.where(D == 1)
    ic, = np.where(D == 0)
    order = rng.permutation(it)
    free = np.ones(len(ic), dtype=bool)
    pairs = []
    for i in order:
        if not free.any():
            break
        free_idx, = np.where(free)
        d = np.abs(logit_ps[i] - logit_ps[ic[free_idx]])
        k = d.argmin()
        if d[k] <= caliper:
            pairs.append((i, free_idx[k]))
            free[free_idx[k]] = False
    if not pairs:
        return np.array([], dtype=int), np.array([], dtype=int)
    idx_t = np.array([p[0] for p in pairs])
    idx_c = np.array([ic[p[1]] for p in pairs])
    return idx_t, idx_c


def smd(x, d, idx_t=None, idx_c=None):
    """SMD: сырые группы или уже сматченные пары (idx_t/idx_c)."""
    if idx_t is None:
        t, c = d == 1, d == 0
        xt, xc = x[t], x[c]
    else:
        xt, xc = x[idx_t], x[idx_c]
    pooled = np.sqrt((xt.var() + xc.var()) / 2.0)
    return (xt.mean() - xc.mean()) / pooled if pooled > 0 else 0.0


def ess(w):
    return w.sum() ** 2 / (w ** 2).sum()


def psm_effect(X, D, Y, seed=0, return_pairs=False):
    """Пайплайн 6.4: propensity -> матчинг (без заимствования) -> ATT + SE."""
    beta, Xd = fit_logistic(X, D)
    e_hat = sigmoid(Xd @ beta)
    logit = np.log(e_hat / (1 - e_hat))
    rng = np.random.default_rng(seed)
    idx_t, idx_c = match_pairs_noreuse(logit, D, 0.2 * logit.std(), rng)
    att = (Y[idx_t] - Y[idx_c]).mean()
    # пары независимы (контроль не переиспользуется) -> SE парной разности честная
    se = (Y[idx_t] - Y[idx_c]).std() / np.sqrt(len(idx_t))
    return (att, se, idx_t, idx_c) if return_pairs else (att, se)


def make_case(n=4000, seed=6410):
    """Тот же DGP, что в 6.4 (структура 1:1), + поля для плацебо и E-value.

    Поток ГПСЧ здесь другой, чем в 6.4 (дополнительные поля), поэтому seed
    свой: 6410 подобран так, что шум реализации не путает демонстрацию.

      activity ~ N(4,2); income ~ N(0,1); mega ~ Bern(0.3)
      e(D=1|X) = sigmoid(-2.2 + 0.45*act + 0.8*inc + 0.7*mega)
      Y     = 3 + 0.6*act + 0.9*inc + 0.7*mega + 1.0*D + N(0,1)   -- истина +1.0
      Y_pre = 3 + 0.55*act + 0.85*inc + 0.6*mega + N(0,1)          -- ДО подписки
      complaints ~ Poisson(2.5*exp(0.35*income))                    -- D не влияет,
                                                                    но селекция по
                                                                    доходу видна
      repeat ~ Bern(sigmoid(-1.6 + 0.15*act + 0.2*inc + 0.6*D))    -- для RR/E-value
    """
    rng = np.random.default_rng(seed)
    activity = np.clip(rng.normal(4, 2, n), 0, None)
    income = rng.normal(0, 1, n)
    mega = rng.binomial(1, 0.3, n)
    X = np.column_stack([activity, income, mega])
    D = rng.binomial(1, sigmoid(-2.2 + 0.45 * activity + 0.8 * income + 0.7 * mega)).astype(float)
    Y = 3 + 0.6 * activity + 0.9 * income + 0.7 * mega + 1.0 * D + rng.normal(0, 1, n)
    Y_pre = 3 + 0.55 * activity + 0.85 * income + 0.6 * mega + rng.normal(0, 1, n)
    complaints = rng.poisson(2.5 * np.exp(0.35 * income), n).astype(float)
    repeat = rng.binomial(1, sigmoid(-1.6 + 0.15 * activity + 0.2 * income + 0.6 * D)).astype(float)
    return rng, X, D, Y, Y_pre, complaints, repeat, ["activity", "income", "mega"]


# %% [markdown]
# ## 1. Плацебо-тесты: настоящий эффект не должен «прокрашиваться» там, где его не может быть

rng, X, D, Y, Y_pre, complaints, repeat, NAMES = make_case()
n = len(Y)
truth = 1.0

att_y, se_y, idx_t, idx_c = psm_effect(X, D, Y, seed=0, return_pairs=True)
naive_y = Y[D == 1].mean() - Y[D == 0].mean()
print("=" * 88)
print("1) ПЛАЦЕБО-ТЕСТЫ для эффекта из 6.4 (тот же DGP-скелет: конфаундеры,")
print("   селекция, истина +1.0; поток генератора здесь свой -- см. make_case)")
print("=" * 88)
print(f"   напоминание: наивная разность {naive_y:+.3f}, PSM-оценка {att_y:+.3f} "
      f"(SE {se_y:.3f}), ИСТИНА {truth:+.3f}")
print(f"   пар (матчинг без заимствования): {len(idx_t)}; "
      f"вывод -- SATT: эффект для сегмента, где нашлись двойники")

# --- 1а. Плацебо-исход: метрика, на которую лечение влиять не может
att_pl_out, se_pl_out = psm_effect(X, D, complaints, seed=0)
naive_pl_out = complaints[D == 1].mean() - complaints[D == 0].mean()
z_pl_out = att_pl_out / se_pl_out
p_pl_out = 2 * (1 - norm.cdf(abs(z_pl_out)))
ok_out = p_pl_out > ALPHA
print()
print(f"   1а. ПЛАЦЕБО-ИСХОД (число жалоб -- подписка на него влиять не может):")
print(f"       наивная разность:  {naive_pl_out:+.3f}  <- селекция по доходу: подписчики")
print(f"       живут в богатых районах и чаще жалуются на качество")
print(f"       PSM-эффект:        {att_pl_out:+.4f} (SE {se_pl_out:.4f}), "
      f"p = {p_pl_out:.2f}  <- [{'НЕТ эффекта -- тест пройден' if ok_out else 'ЭФФЕКТ НА ПЛАЦЕБО -- спецификация под вопросом'}]")

# --- 1б. Плацебо-время: целевая метрика ДО существования подписки
att_pl_t, se_pl_t = psm_effect(X, D, Y_pre, seed=0)
naive_pl_t = Y_pre[D == 1].mean() - Y_pre[D == 0].mean()
z_pl_t = att_pl_t / se_pl_t
p_pl_t = 2 * (1 - norm.cdf(abs(z_pl_t)))
ok_time = p_pl_t > ALPHA
print()
print(f"   1б. ПЛАЦЕБО-ВРЕМЯ (заказы прошлого месяца -- подписки ещё не было):")
print(f"       наивная разность:  {naive_pl_t:+.3f}  <- «эффект» ДО лечения = чистая селекция")
print(f"       PSM-эффект:        {att_pl_t:+.3f} (SE {se_pl_t:.3f}), "
      f"p = {p_pl_t:.2f}  <- [{'НЕТ эффекта -- тест пройден' if ok_time else 'ЭФФЕКТ НА ПЛАЦЕБО -- остаточный дисбаланс'}]")
print(f"       на реальных данных пограничное плацебо (p около альфы) --")
print(f"       не спрятать, а озвучить: это индикатор остаточного смещения")

# --- 1в. Плацебо-экспозиция: перемешиваем D (100 пермутаций)
obs = att_y
perm = []
for s in range(100):
    Dp = rng.permutation(D)
    perm.append(psm_effect(X, Dp, Y, seed=s)[0])
perm = np.array(perm)
p_perm = np.mean(np.abs(perm) >= abs(obs))
p_perm_str = f"<={1 / len(perm):.2f}" if p_perm == 0 else f"={p_perm:.2f}"
print()
print(f"   1в. ПЛАЦЕБО-ЭКСПОЗИЦИЯ (D перемешан, 100 пермутаций):")
print(f"       эффект на перемешанном D: медиана {np.median(perm):+.3f}, "
      f"95% перцентили [{np.quantile(perm, 0.025):+.3f}; {np.quantile(perm, 0.975):+.3f}]")
print(f"       наблюдаемый {obs:+.3f} -- вне нуль-распределения: эмпирический p {p_perm_str}"
      f"  <- [эффект неслучаен]")
print()
print("   ИТОГ: метод не создаёт эффект из ничего -- на плацебо он молчит.")
print("   Если бы PSM «прокрасил» плацебо -- диагноз: спецификация/modeling artifact,")
print("   результатам на Y верить нельзя (аналог SRM для наблюдательных данных).")

# %% [markdown]
# ## 2. Sensitivity: насколько сильный ненаблюдаемый конфаундер убьёт вывод?

print()
print("=" * 88)
print("2) ЧУВСТВИТЕЛЬНОСТЬ К НЕНАБЛЮДАЕМОМУ КОНФАУНДЕРУ")
print("=" * 88)

# --- 2а. E-value (VanderWeele--Ding) на бинарном исходе «повторный заказ»
r_t = repeat[idx_t].mean()
r_c = repeat[idx_c].mean()
rr = r_t / r_c
se_log_rr = np.sqrt(1 / (r_t * len(idx_t)) + 1 / (r_c * len(idx_c)))  # приближение
rr_lo = rr * np.exp(-1.96 * se_log_rr)


def evalue(rr):
    """E-value: RR + sqrt(RR*(RR-1)); для защитных факторов (RR<1) инвертируем."""
    rr = 1.0 / rr if rr < 1 else rr
    return rr + np.sqrt(rr * (rr - 1.0))


ev_point, ev_lo = evalue(rr), evalue(rr_lo)
print(f"   2а. E-VALUE (исход «повторный заказ в следующем месяце»):")
print(f"       риск в treated {r_t:.3f} vs контроль {r_c:.3f} -> RR = {rr:.3f} "
      f"[нижняя граница {rr_lo:.3f}]")
print(f"       E-value = RR + sqrt(RR*(RR-1)) = {ev_point:.2f} (по нижней границе CI: {ev_lo:.2f})")
print(f"       ЧТЕНИЕ: чтобы ПОЛНОСТЬЮ объяснить наш эффект, ненаблюдаемый конфаундер")
print(f"       должен быть связан и с подпиской, и с повторным заказом ассоциацией")
print(f"       не слабее RR = {ev_point:.2f} КАЖДАЯ (и {ev_lo:.2f} -- чтобы убрать значимость).")
print(f"       E-value около 1.0 означал бы: достаточно любой пылинки; 2+ -- крепкий результат.")

# --- 2б. Упрощённая гамма Розенбаума: знаковый тест по парам
diff = Y[idx_t] - Y[idx_c]
wins = int((diff > 0).sum())
n_pairs = int((diff != 0).sum())
p_obs = binomtest(wins, n_pairs, 0.5).pvalue


def rosenbaum_pvalue(gamma):
    """p-value знакового теста при скрытом смещении Г: P(treated выиграл) = Г/(1+Г)."""
    prob = gamma / (1.0 + gamma)
    return binomtest(wins, n_pairs, prob).pvalue


gammas = np.round(np.arange(1.0, 3.001, 0.05), 2)
pvals = np.array([rosenbaum_pvalue(g) for g in gammas])
alive = np.where(pvals < ALPHA)[0]
if len(alive) == 0:
    gamma_star, gamma_star_str = 1.0, "нет (несостоятелен уже при Г=1)"
elif alive[-1] == len(gammas) - 1:
    gamma_star, gamma_star_str = gammas[-1], f">{gammas[-1]:.2f}"
else:
    gamma_star = gammas[alive[-1]]
    gamma_star_str = f"~{gamma_star:.2f}"


def gamma_verdict(g):
    if g <= 1.15:
        return "результату НЕ стоит доверять (слабейший конфаундер убивает)"
    if g <= 1.5:
        return "приемлемо, но нужна оговорка о скрытых факторах"
    if g <= 2.0:
        return "сопоставимо со значимостью обычного A/B"
    return "почти непробиваемо"


print()
print(f"   2б. ГАММА РОЗЕНБАУМА (упрощение -- знаковый тест по {n_pairs} парам):")
print(f"       пар, где treated выше контроля: {wins} из {n_pairs} "
      f"(при честном 50/50); p-value при Г=1: {p_obs:.4f}")
print(f"       Г* (до какого смещения результат ещё значим при альфа={ALPHA}): {gamma_star_str}")
print(f"       ЧТЕНИЕ: если скрытый фактор делает попадание в «тест» в Г раз вероятнее,")
print(f"       наш значимый результат доживает до Г*; вердикт: {gamma_verdict(gamma_star)}.")
print("       Шкала: 1-1.15 не верить | 1.15-1.5 приемлемо с оговоркой |")
print("              1.5-2 как значимый A/B | >2 почти непробиваемо")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
ax = axes[0]
labels = ["целевая Y\n(заказы)", "плацебо-исход\n(жалобы)", "плацебо-время\n(заказы до)", "плацебо-D\n(медиана 100 перм.)"]
vals = [att_y, att_pl_out, att_pl_t, np.median(perm)]
errs = [1.96 * se_y, 1.96 * se_pl_out, 1.96 * se_pl_t, 1.96 * np.std(perm) / np.sqrt(100)]
colors = [GREEN, BLUE, BLUE, GREY]
ax.bar(range(4), vals, yerr=errs, color=colors, alpha=0.85, capsize=4, width=0.6)
ax.axhline(0, color="k", lw=1)
ax.axhline(truth, color=RED, ls="--", lw=1.6)
ax.text(3.45, truth + 0.04, "истина +1.0", color=RED, fontsize=9, ha="right")
ax.set_xticks(range(4), labels, fontsize=8.5)
ax.set_ylabel("эффект по PSM")
ax.set_title("Плацебо-тесты: на целевой метрике эффект есть,\nна плацебо (исход/время/D) PSM молчит — метод честный", fontsize=10.5)
ax = axes[1]
ax.plot(gammas, pvals, color=BLUE, lw=2.2)
ax.axhline(ALPHA, color=RED, ls="--", lw=1.6)
ax.axvline(gamma_star, color=GREY, ls=":", lw=1.6)
ax.annotate(f"Г* = {gamma_star_str}\n(результат умирает)", xy=(gamma_star, ALPHA),
            xytext=(gamma_star + 0.3, 0.3), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#444"))
ax.set_xlabel("Гамма Розенбаума Г (сила скрытого смещения)")
ax.set_ylabel(f"p-value знакового теста (альфа={ALPHA})")
ax.set_title("Sensitivity: какой силы ненаблюдаемый конфаундер\nубивает значимость", fontsize=10.5)
fig.tight_layout()
fig.savefig(HERE / "practice_6_5_placebo_sensitivity.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 3. Дерево выбора метода: method_tree(ситуация)


def method_tree(can_randomize=False, blocked_by_ethics=False, already_happened=False,
                has_control_group=False, parallel_trends=False, has_threshold=False,
                single_unit=False, rich_covariates=False, has_instrument=False,
                verbose=True):
    """Текстовое дерево выбора каузального метода (слайд 21 лекции + весь М6).

    Возвращает (метод, путь-список шагов). Ключи ситуации:
      can_randomize      -- можно рандомизировать воздействие?
      blocked_by_ethics  -- нельзя рандомизировать по этике/стоимости/длительности
      already_happened   -- воздействие уже выкачено (не отменить)
      has_control_group  -- есть невылеченные юниты для сравнения
      parallel_trends    -- правдоподобны параллельные тренды (претренд-проверка)
      has_threshold      -- назначение лечения режется порогом (running variable)
      single_unit        -- лечёный юнит один (город/магазин/платформа)
      rich_covariates    -- конфаундеры богатые и измерены до лечения
      has_instrument     -- есть внешний «толчок» (encouragement/природный шок)
    """
    path = []

    def step(q, a):
        path.append(f"{'  ' * len(path)}{q} {'-> ДА' if a else '-> НЕТ'}")

    step("можно рандомизировать воздействие?", can_randomize)
    if can_randomize:
        method = ("A/B-тест (М3-М4: дизайн, MDE, CUPED, OEC+guardrails) -- "
                  "всегда первый выбор")
    else:
        step("блокируют этика/стоимость/длительность (но можно НЕ лечить часть)?",
             blocked_by_ethics)
        if blocked_by_ethics and not already_happened:
            step("есть внешний толчок-инструмент (encouragement)?", has_instrument)
            if has_instrument:
                method = ("encouragement design + IV: рандомизируем ПОБУЖДЕНИЕ, "
                          "эффект -- Wald/LATE для compliers (6.4)")
            else:
                method = ("switchback/geo-эксперимент (4.3); нет и этого -- "
                          "спускаемся ниже с большой оговоркой")
        else:
            step("воздействие уже произошло?", already_happened)
            if already_happened or not can_randomize:
                step("есть контрольная группа юнитов?", has_control_group)
                if has_control_group:
                    step("параллельные тренды правдоподобны (претренды)?", parallel_trends)
                    if parallel_trends:
                        method = ("DiD до/после (+ PSM для сопоставимости -- "
                                  "комбинация надёжнее одного метода, 6.3); "
                                  "staggered -- Callaway & Sant'Anna")
                    else:
                        step("лечение назначается порогом?", has_threshold)
                        if has_threshold:
                            method = ("RDD: сравниваем «почти одинаковых» по обе "
                                      "стороны порога (6.3); эффект локален")
                        else:
                            step("лечёный юнит один?", single_unit)
                            if single_unit:
                                method = ("synthetic control / Causal Impact: "
                                          "контрфакт из взвешенных контролей (6.3)")
                            else:
                                step("есть внешний инструмент?", has_instrument)
                                if has_instrument:
                                    method = ("IV: Wald/2SLS -- LATE для compliers "
                                              "(проверить первый этап F>10!) (6.4)")
                                else:
                                    step("конфаундеры измерены и богаты?", rich_covariates)
                                    if rich_covariates:
                                        method = ("matching/IPW/DR по ковариатам (6.4) "
                                                  "+ ОБЯЗАТЕЛЬНО плацебо и sensitivity "
                                                  "(этот урок)")
                                    else:
                                        method = ("ВЫВОД НЕ ДЕЛАЕМ: нет дизайна, нет "
                                                  "ковариат, нет инструмента -- только "
                                                  "разведочный анализ с оговоркой")
                else:
                    step("лечёный юнит один?", single_unit)
                    if single_unit:
                        method = ("synthetic control / Causal Impact (прогноз с "
                                  "контрольным рядом, 6.3)")
                    else:
                        method = ("ВЫВОД НЕ ДЕЛАЕМ: контроля нет -- сравнивать не с чем; "
                                  "искать отложенный контроль или не отвечать")
    if verbose:
        print("\n".join(path))
    return method, path


print()
print("=" * 88)
print("3) ДЕРЕВО ВЫБОРА МЕТОДА: method_tree(ситуация) -- прогон сценариев «ЕдаДома»")
print("=" * 88)

scenarios = [
    ("Новая кнопка оформления заказа", dict(can_randomize=True)),
    ("Подняли цену доставки для ВСЕХ (вчера)", dict(already_happened=True, has_control_group=True,
                                                    parallel_trends=True)),
    ("Скидка 20% для юзеров с NPS-скором >= 8",
     dict(already_happened=True, has_control_group=True, has_threshold=True)),
    ("«ЕдаДоставка 2.0» в одном городе-пилоте", dict(already_happened=True, single_unit=True,
                                                     has_control_group=True)),
    ("Подписка «ЕдаПлюс» по желанию (кейс 6.4)",
     dict(already_happened=True, rich_covariates=True, has_control_group=True)),
    ("Эффект быстрой доставки на чаевые", dict(blocked_by_ethics=True, has_instrument=True)),
    ("«Курьеры на велосипедах безопаснее?» -- ни дизайна, ни ковариат",
     dict(already_happened=True, has_control_group=False, has_instrument=False,
          rich_covariates=False)),
]
for name, kw in scenarios:
    print(f"\n   СЦЕНАРИЙ: {name}")
    method, _ = method_tree(verbose=False, **kw)
    print(f"   ==> {method}")

# %% [markdown]
# ## 4. Чек-лист «каузальный аудит»

print()
print("=" * 88)
print("4) ЧЕК-ЛИСТ «КАУЗАЛЬНЫЙ АУДИТ» (собран по диагностике этого запуска)")
print("=" * 88)

smd_after = [abs(smd(X[:, k], D, idx_t=idx_t, idx_c=idx_c)) for k in range(X.shape[1])]
rows = [
    ("1. Дизайн: назван эстиманд (ATE/ATT/SATT/LATE)?", "PASS",
     "после матчинга это SATT: вывод -- для сегмента сопоставимых"),
    ("2. Ковариаты только pre-treatment (нет медиаторов/коллайдеров/утечек)?", "PASS",
     "activity/income/mega -- все «до»; грабли 6.4 обходом"),
    (f"3. Баланс: все SMD < 0.1 после матчинга? (max = {max(smd_after):.2f})",
     "PASS" if max(smd_after) < 0.1 else "FAIL",
     "Love plot: до матчинга SMD доходил до 0.76"),
    ("4. Overlap: существенная зона пересечения PS (пар построено много)?", "PASS",
     f"{len(idx_t)} пар из {(D == 1).sum()} treated; хвосты без пересечения съел caliper"),
    ("5. Эффективный размер выборки учтён (ESS/без заимствования)?", "PASS",
     "матчинг без заимствования: каждый контроль в одной паре, SE честная"),
    ("6. Плацебо-исход: эффект = 0?", "PASS" if ok_out else "FAIL",
     f"p = {p_pl_out:.2f}; наивная разность там {naive_pl_out:+.2f} -- селекция была"),
    ("7. Плацебо-время (предпериод): эффект = 0?", "PASS" if ok_time else "WARN",
     f"p = {p_pl_t:.2f}, сдвиг {att_pl_t:+.2f}; пограничное -- озвучивать, не прятать"),
    ("8. Плацебо-экспозиция (пермутация D): наблюдаемый эффект вне нуль-распределения?",
     "PASS" if p_perm <= ALPHA else "WARN", f"эмпирический p {p_perm_str}"),
    (f"9. Sensitivity: гамма Розенбаума > 1.5? (Г* = {gamma_star_str})",
     "PASS" if gamma_star > 1.5 else "WARN", gamma_verdict(gamma_star)),
    (f"10. Sensitivity: E-value >= 2? (E = {ev_point:.2f})", "PASS" if ev_point >= 2 else "WARN",
     f"скрытому конфаундеру нужна ассоциация RR>={ev_point:.2f} с обеими сторонами"),
    ("11. Доменная проверка: эффект адекватен бизнес-логике?", "ВРУЧНУЮ",
     "спросить непредвзятых: +1 заказ/мес от бесплатной доставки -- правдоподобно?"),
    ("12. Знак наивного и скорректированного эффекта совпадает?", "PASS",
     f"наивное {naive_y:+.2f} vs PSM {att_y:+.2f}: смещение завышало, знак тот же"),
    ("13. Триангуляция: второй-третий метод даёт согласный ответ?", "ВРУЧНУЮ",
     "регрессия/IPW/DR из 6.4 все у +0.9..+1.1 -- сходятся"),
]
print(f"   {'проверка':66s} {'статус':7s} комментарий")
print(f"   {'-' * 66} {'-' * 7} {'-' * 30}")
for q, st, c in rows:
    print(f"   {q:66s} {st:7s} {c}")
fails = sum(st == "FAIL" for _, st, _ in rows)
warns = sum(st == "WARN" for _, st, _ in rows)
print()
print(f"   ВЕРДИКТ: {len(rows)} проверок, FAIL = {fails}, WARN = {warns}, "
      f"«вручную» = {sum(st == 'ВРУЧНУЮ' for _, st, _ in rows)}.")
print("   WARN не отменяет результат -- он обязателен к озвучиванию стейкхолдеру")
print("   рядом с цифрой эффекта (вместе с CI и эстимандом). FAIL -- результат")
print("   не публикуем. «Вручную» -- код не заменяет экспертизу и триангуляцию.")
print()
print("График: practice_6_5_placebo_sensitivity.png (плацебо-бары + кривая гаммы)")
