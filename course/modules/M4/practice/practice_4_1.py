# -*- coding: utf-8 -*-
"""Практика 4.1 — Цикл эксперимента: дизайн-док, выбор OEC, «⅓ идей выигрывают»
(сквозной кейс «ЕдаДома», урок 4.1 модуля «АБ-тесты на практике»).

Принцип курса: не верь интуиции — проверь симуляцией.

Что делаем:
1) Генератор дизайн-дока: функция принимает гипотезу (X -> Y -> Z), метрики
   (OEC + guardrails + invariants) и MDE -> выдаёт n, даты, decision rule,
   ramp-план — готовый документ до запуска теста;
2) симуляция «⅓ идей выигрывают» (Kohavi, KDD'13): бэклог идей и три
   стратегии — катить всё / тестировать всё / тестировать по приоритету;
3) выбор OEC из кандидатов по чувствительности: SE и MDE каждой метрики
   на синтетике «ЕдаДома», карта «чувствительность × близость к деньгам».

Запуск из корня репозитория: python3 course/modules/M4/practice/practice_4_1.py
Графики сохраняются в папку images/ этого модуля.
"""

# %% [markdown]
# Шаг 0. Импорты и константы «ЕдаДома»: α=5%, мощность 80%, трафик
# 1 500 юзеров/будень и 2 100/выходной (неделя = 11 700, урок 3.1).

# %%
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

RNG = np.random.default_rng(20260910)
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

ALPHA, POWER = 0.05, 0.80
Z_A = stats.norm.ppf(1 - ALPHA / 2)  # 1.96
Z_B = stats.norm.ppf(POWER)  # 0.84
WEEKDAY, WEEKEND = 1_500, 2_100


def n_per_group(mde, sigma, alpha=ALPHA, power=POWER):
    """n на группу: двухвыборочная формула урока 3.1 (с множителем 2)."""
    z1 = stats.norm.ppf(1 - alpha / 2)
    z2 = stats.norm.ppf(power)
    return 2 * (z1 + z2) ** 2 * sigma ** 2 / mde ** 2


def oec_rule_power(n, p0, effect, business_min=0.005, alpha=0.05):
    """Нормальное приближение: обнаружить ненулевой эффект / пройти порог OEC.

    Второе число не учитывает дополнительные условия безопасности guardrails,
    поэтому это верхняя оценка вероятности полного решения SHIP.
    """
    p1 = p0 + effect
    se = math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n)
    z = stats.norm.isf(alpha / 2)
    detect = stats.norm.sf(z - effect / se) + stats.norm.cdf(-z - effect / se)
    practical = stats.norm.sf(z - (effect - business_min) / se)
    return float(detect), float(practical)


# %% [markdown]
# Шаг 1. Генератор дизайн-дока. Ядро курса: всё, что решает судьбу теста,
# фиксируется ДО запуска. Функция собирает документ из гипотезы X->Y->Z,
# метрик и MDE: считает n (формула 3.1), длительность целыми неделями,
# даты через pandas, decision rule 2x2 и ramp-план.

# %%
def decision_rule(oec, guardrails, business_min=0.005, alpha=ALPHA):
    """Предзаданная политика по CI, бизнес-порогу и границам безопасности."""
    return [
        "DECISION RULE (пишется ДО запуска):",
        f"  Бизнес-порог для {oec}: +{business_min*100:.2f} п.п.; он отличается от MDE.",
        f"  SHIP: нижняя граница {1-alpha:.0%} CI выше бизнес-порога;",
        "        односторонние границы всех guardrails исключают вред выше допустимого margin.",
        "  ROLLBACK: CI OEC целиком ниже 0 либо подтверждён неприемлемый вред.",
        "  HOLD: остальные случаи, в том числе не доказанная безопасность.",
        "  Границы guardrails и поправка на их число фиксируются до старта:",
        *[f"        {guardrail}" for guardrail in guardrails],
        "  Не добираем данные до значимости. Post-hoc сегменты — гипотезы",
        "  для новой проверки; они не заменяют первичный вердикт этого теста.",
    ]


def build_design_doc(number, name, hyp, oec, oec_p0, mde, guardrails,
                     start="2026-10-05", alpha=ALPHA, power=POWER,
                     weekday=WEEKDAY, weekend=WEEKEND, owner="продакт каталога",
                     business_min=0.005, window_days=14):
    """Гипотеза X->Y->Z + метрики + MDE -> готовый дизайн-док теста.

    Возвращает (текст документа, словарь расчётных параметров).
    """
    sigma = math.sqrt(oec_p0 * (1 - oec_p0))  # бинарная OEC: sigma^2 = p(1-p)
    n_raw = n_per_group(mde, sigma, alpha, power)
    n = math.ceil(n_raw * 1.05)  # запас 5% (урок 3.1)
    weekly = 5 * weekday + 2 * weekend
    weeks = max(2, math.ceil(2 * n / weekly))  # целыми неделями, не меньше 2
    t0 = pd.Timestamp(start)
    enrollment_end = t0 + pd.Timedelta(days=7 * weeks)  # правая граница, не включается
    analysis_date = enrollment_end + pd.Timedelta(days=window_days)
    detect_power, practical_power = oec_rule_power(n, oec_p0, mde, business_min, alpha)
    expected_n = weekly * weeks / 2
    calendar_detect, calendar_practical = oec_rule_power(expected_n, oec_p0, mde, business_min, alpha)

    lines = [
        "=" * 79,
        f"ДИЗАЙН-ДОК №{number}: «{name}»",
        "=" * 79,
        "ГИПОТЕЗА (потому что X -> изменится Y -> выиграет Z):",
        f"  X: {hyp['x']}",
        f"  Y: {hyp['y']}",
        f"  Z: {hyp['z']}",
        "",
        "МЕТРИКИ (роли определены до запуска, урок 4.1):",
        f"  OEC (метрика решения): {oec} (p0 = {oec_p0:.1%} по пре-периоду)",
        "  Guardrails (не должны просесть): " + "; ".join(guardrails),
        "  Инварианты: treatment не должен их менять; случайные отклонения возможны.",
        "      pre-treatment признаки и SRM назначения; уровень тревоги фиксируется заранее",
        "  Отдельная диагностика: полнота и задержка логов; продукт может менять число событий",
        "",
        "ЮНИТ рандомизации = юнит анализа = user_id (урок 3.2)",
        "",
        f"СТАТИСТИКА: alpha={alpha:.0%}, целевая мощность против H0: эффект=0 — {power:.0%}, "
        f"MDE=+{mde * 100:.1f} п.п. (+{mde / oec_p0:.1%} отн.)",
        f"  sigma^2 = p0(1-p0) = {oec_p0 * (1 - oec_p0):.4f}",
        f"  n = 2*({stats.norm.isf(alpha/2):.2f}+{stats.norm.ppf(power):.2f})^2*sigma^2/mde^2 = {n_raw:,.0f}"
        f" -> с запасом 5%: {n:,} юзеров/группу",
        f"  При минимальном n и истинном эффекте MDE: обнаружение отличия от 0 ≈{detect_power:.0%};",
        f"  нижняя граница CI выше бизнес-порога ≈{practical_power:.0%} (только OEC).",
        "  Вероятность SHIP не выше последнего числа: дополнительно нужны безопасные guardrails.",
        "  Если нужна заданная вероятность SHIP, проектируем объём под бизнес-порог и guardrails отдельно.",
        "",
        f"НАБОР: {weeks} полные недели, [{t0.date()}, {enrollment_end.date()}),"
        f" набираем ~{weekly * weeks:,} новых уникальных юзеров",
        f"  Округление до недель даёт ≈{expected_n:,.0f}/руку: detection ≈{calendar_detect:.0%}, бизнес-порог OEC ≈{calendar_practical:.0%}.",
        f"  Метрика каждого юзера созревает {window_days} дней; финальный анализ не раньше {analysis_date.date()}.",
        "  Дата старта — начало полного теста после ramp; трафик выше относится именно к нему.",
        "КРИТЕРИЙ ОСТАНОВКИ: fixed horizon после созревания последней когорты; в процессе — только",
        "  invariants и guardrails (ранняя остановка по вреду); последовательные",
        "  схемы — только заранее выбранные (урок 3.7)",
        "",
        *decision_rule(oec, guardrails, business_min, alpha),
        "",
        "RAMP: 1% (1 день, canary: краши/латентность) -> 5% (2 дня, guardrails)",
        "  -> 50% (полный тест: с этого момента копим n) -> 100% после решения",
        f"ВЛАДЕЛЕЦ РЕШЕНИЯ: {owner}",
        "=" * 79,
    ]
    meta = dict(n=n, weeks=weeks, t0=t0, enrollment_end=enrollment_end,
                analysis_date=analysis_date, sigma=sigma,
                detect_power=detect_power, practical_power=practical_power,
                expected_n=expected_n, calendar_practical_power=calendar_practical)
    return "\n".join(lines), meta


# %% [markdown]
# Два теста из бэклога «ЕдаДома»: разный OEC -> разный n и разные даты.

# %%
doc1, m1 = build_design_doc(
    "ED-2026-114", "Умная сортировка главной по времени доставки",
    hyp={"x": "юзеры не находят самые быстрые рестораны (медиана поиска 4.2 мин, "
              "31% сессий не доходит до карточки)",
         "y": "сортировка по ETA поднимет конверсию в заказ (окно 2 недели)",
         "z": "больше заказов -> больше выручки (заказы — драйвер, но итоговая ценность зависит также от чека и маржи)"},
    oec="конверсия в заказ (2 нед.)", oec_p0=0.23, mde=0.015,
    guardrails=["p95 латентности списка (+<100 мс)", "отписки от пушей (+<5% отн.)",
                "доля отмен заказов (+<0.3 п.п.)"],
)
doc2, m2 = build_design_doc(
    "ED-2026-115", "Пуш-напоминание о брошенной корзине",
    hyp={"x": "72% корзин бросаются после добавления десерта",
         "y": "пуш со скидкой на брошенную корзину вернёт часть юзеров",
         "z": "дополнительные заказы без каннибализации органических"},
    oec="доля открытий пуша (диагностика механики, не итоговая ценность)", oec_p0=0.062, mde=0.01,
    guardrails=["отписки от пушей (+<5% отн.)", "деинсталлы (+<0.2 п.п.)",
                "crash-free rate (снижение не более 0.1 п.п.)"],
    owner="продакт удержания",
)
print(doc1)
print()
print(doc2)
print("Документ 2 проверяет механику. Для решения о релизе дополнительно нужен тест маржи/пользователя.")
print()
print(f"Справка из урока 3.1: недельная конверсия (p0=12%, MDE +1 п.п. = +8.3% отн.) "
      f"требовала 16 577 юзеров; окно 2 недели (p0=23%, то же +8.3% отн. = +1.9 п.п.) "
      f"обходится в {n_per_group(0.019, math.sqrt(0.23 * 0.77)):,.0f} — окно метрики "
      f"тоже инструмент чувствительности.")
print(f"А пуш из док-2 целится в +16.1% отн. и потому довольствуется 9 586 юзерами;"
      f"\nна те же +8.3% отн. редкой метрики (p0=6.2%) понадобилось бы "
      f"{n_per_group(0.083 * 0.062, math.sqrt(0.062 * 0.938)):,.0f} юзеров на группу.")

# %% [markdown]
# Шаг 2. «⅓ идей выигрывают» (Kohavi, KDD'13): в зрелых компаниях лишь около
# трети идей улучшают целевые метрики, треть нейтральны, треть вредят.
# Квартал: 24 идеи, мощность платформы — 12 тестов (6 двухнедельных волн ×
# 2 параллельных слота — как устроены слои, увидим в уроке 4.2).
# Три стратегии: катить всё без тестов / тестировать случайные / тестировать
# по приоритету (зашумлённая оценка команды коррелирует с истиной).
# «Выигрышная» идея: истинный эффект OEC > +0.5% (порог окупаемости).

# %%
N_IDEAS, CAPACITY, WORLDS = 24, 12, 400
WIN_T = 0.5  # %: бизнес-порог «идея окупается»
SE_TEST = 2.0 / (Z_A + Z_B)  # шум теста: истинный эффект +2% ловится с ~80%

STRATS = ("катим всё без тестов", "тестируем случайные 12", "тестируем топ-12 по приоритету")
cum_mean = {s: np.zeros(N_IDEAS) for s in STRATS}
finals = {s: [] for s in STRATS}
false_ship = {s: [] for s in STRATS}
ship_share = {s: [] for s in STRATS}
share_win = []

for _ in range(WORLDS):
    k = N_IDEAS // 3
    true_eff = np.concatenate([RNG.normal(2.2, 0.6, k),   # ⅓ выигрышных
                               RNG.normal(0.0, 0.4, k),   # ⅓ нейтральных
                               RNG.normal(-2.2, 0.6, k)])  # ⅓ вредных
    RNG.shuffle(true_eff)
    prior = true_eff + RNG.normal(0.0, 1.5, N_IDEAS)  # оценка команды
    obs = true_eff + RNG.normal(0.0, SE_TEST, N_IDEAS)  # результат теста
    ship = (obs > 0) & (obs / SE_TEST > Z_A)  # значимый плюс -> катим

    share_win.append(np.mean(true_eff > WIN_T))

    res = {}
    # S0: катим всё, порядок случайный (24 решения)
    o0 = RNG.permutation(N_IDEAS)
    res[STRATS[0]] = (true_eff[o0].copy(), true_eff[o0], np.ones(len(o0), dtype=bool))
    # S1: тестируем 12 случайных идей, катим «значимые плюсы»
    o1 = RNG.permutation(N_IDEAS)[:CAPACITY]
    res[STRATS[1]] = (np.where(ship[o1], true_eff[o1], 0.0), true_eff[o1], ship[o1])
    # S2: тестируем топ-12 по приоритету
    o2 = np.argsort(-prior)[:CAPACITY]
    res[STRATS[2]] = (np.where(ship[o2], true_eff[o2], 0.0), true_eff[o2], ship[o2])

    for s, (value, tested_true, shipped) in res.items():
        cum = np.cumsum(value)
        cum_mean[s][: len(cum)] += cum / WORLDS
        finals[s].append(cum[-1])
        false_ship[s].append(int(np.sum(shipped & (tested_true < 0))))
        ship_share[s].append(int(np.sum(shipped)))

print("=" * 78)
print("2) «⅓ ИДЕЙ ВЫИГРЫВАЮТ»: квартал, %d идей, платформа тянет %d тестов" % (N_IDEAS, CAPACITY))
print("-" * 78)
print(f"   доля идей с эффектом выше порога окупаемости (+{WIN_T}%): "
      f"{np.mean(share_win):.1%} (Kohavi: около 1/3)")
print(f"   {'стратегия':<32} | {'раскачано':>8} | {'вредных из них':>14} | {'итог OEC':>9}")
print(f"   {'-'*32}-+-{'-'*8}-+-{'-'*14}-+-{'-'*9}")
for s in STRATS:
    print(f"   {s:<32} | {np.mean(ship_share[s]):>8.1f} | {np.mean(false_ship[s]):>14.2f} | "
          f"{np.mean(finals[s]):>+8.2f}%")

# %%
fig, ax = plt.subplots(figsize=(8.8, 4.6))
colors = {"катим всё без тестов": "#C44E52", "тестируем случайные 12": "#4C72B0",
          "тестируем топ-12 по приоритету": "#55A868"}
for s in STRATS:
    upto = N_IDEAS if s == STRATS[0] else CAPACITY
    ax.plot(np.arange(1, upto + 1), cum_mean[s][:upto], "-o", ms=4, lw=2.2,
            color=colors[s], label=f"{s}: {np.mean(finals[s]):+.1f}%")
ax.axhline(0, color="gray", lw=1)
ax.axvline(CAPACITY + 0.5, color="gray", ls=":", lw=1.4)
ax.text(CAPACITY + 0.7, np.mean(finals[STRATS[1]]) * 0.45,
        "лимит платформы:\n12 тестов/квартал", fontsize=9, color="gray")
ax.set_xlabel("решение № (хронология квартала)")
ax.set_ylabel("накопленный истинный эффект на OEC, %")
ax.set_title("«⅓ идей выигрывают» (Kohavi, KDD'13): без тестов квартал блуждает около нуля,\n"
             "тесты оставляют выигрыш; приоритизация усиливает итог при той же платформе")
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_4_1_third_wins.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_4_1_third_wins.png")

# %% [markdown]
# Шаг 3. Выбор OEC по чувствительности. Кандидаты для сортировки: CTR
# (открыл карточку ресторана), конверсия в заказ за 2 недели, выручка
# на юзера за 4 недели. Генерируем пре-период 20 000 юзеров «ЕдаДома»:
# у каждого устойчивая «активность» u (общий драйвер) и своя интенсивность
# заказов lambda (сверхразбросанная — у доставки еды покупатели «привычные»).
# Считаем SE разности при n=8 000/группу и MDE; усиливаем кандидатов
# CUPED по пре-периоду (урок 3.5) и винзоризацией P99 (урок 3.4).

# %%
N_PRE, N_GROUP = 20_000, 8_000
u = RNG.normal(0, 1, N_PRE)  # устойчивая активность юзера (общий драйвер)
z_click = RNG.normal(0, 1, N_PRE)  # «кликальная» шумовая компонента

opened = RNG.binomial(1, expit(0.35 + 0.75 * u + 0.66 * z_click))  # CTR, неделя
lam = np.exp(-2.30 + 1.50 * u)  # интенсивность заказов (2 нед.), сверхразброс
orders = RNG.poisson(lam)
cr2 = (orders > 0).astype(float)  # конверсия в заказ, окно 2 недели
pre_orders = RNG.poisson(lam)  # независимое окно пре-периода (та же lambda)
pre_cr2 = (pre_orders > 0).astype(float)
check = RNG.lognormal(math.log(766) - 0.70 ** 2 / 2, 0.70, N_PRE)  # чек заказа
pre_check = RNG.lognormal(math.log(766) - 0.70 ** 2 / 2, 0.70, N_PRE)
revenue = orders * check  # выручка/юзер (2 нед.)
pre_revenue = pre_orders * pre_check  # выручка пре-периода
rev_wins = np.minimum(revenue, np.percentile(revenue, 99))  # винзоризация P99


def cuped(metric, pre):
    """CUPED: metric - theta*(pre - mean(pre)), theta = cov/var (урок 3.5)."""
    theta = np.cov(metric, pre)[0, 1] / np.var(pre, ddof=1)
    return metric - theta * (pre - pre.mean())


rho_cr = np.corrcoef(cr2, pre_cr2)[0, 1]
rho_rev = np.corrcoef(revenue, pre_revenue)[0, 1]
cr_cuped = cuped(cr2, pre_cr2)
rev_cuped = cuped(revenue, pre_revenue)

rows = []
for label, arr, typical in (
        ("CTR: открыл карточку (нед.)", opened, 0.10),
        ("CR в заказ (2 нед.)", cr2, 0.10),
        ("CR + CUPED (пре-CR)", cr_cuped, 0.10),
        ("Выручка/юзер (2 нед.)", revenue, 0.03),
        ("Выручка винзор. P99", rev_wins, 0.03),
        ("Выручка + CUPED (пре-выручка)", rev_cuped, 0.03)):
    mean, sd = arr.mean(), arr.std(ddof=1)
    se = sd * math.sqrt(2 / N_GROUP)
    mde_rel = (Z_A + Z_B) * se / mean
    rows.append({
        "метрика": label,
        "среднее": f"{mean:,.3f}" if mean < 10 else f"{mean:,.0f}",
        "SD": f"{sd:,.3f}" if sd < 10 else f"{sd:,.0f}",
        "CV": f"{sd / mean:.2f}",
        "SE разн.": f"{se:.3f}" if se < 10 else f"{se:.1f}",
        "MDE отн. (n=8k)": f"{mde_rel:.1%}",
        "типичный эффект": f"{typical:.0%}",
        "мощность ≥80%": "да" if mde_rel <= typical else "нет",
        "corr с выручкой": f"{np.corrcoef(arr, revenue)[0, 1]:.2f}",
    })
df_oec = pd.DataFrame(rows).set_index("метрика")

print("=" * 78)
print("3) ВЫБОР OEC ПО ЧУВСТВИТЕЛЬНОСТИ (пре-период %s юзеров, n=%s/группу)" % (f"{N_PRE:,}", f"{N_GROUP:,}"))
print("-" * 78)
print(df_oec.to_string())
print(f"\n   rho(CR, пре-CR) = {rho_cr:.2f};  rho(выручка, пре-выручка) = {rho_rev:.2f}")
print(f"   CUPED сжимает SE: CR на {1 - math.sqrt(1 - rho_cr ** 2):.0%}, "
      f"выручку на {1 - math.sqrt(1 - rho_rev ** 2):.0%} (урок 3.5)")
print("   Мощность ≥80% — приближённое сравнение MDE с заданным эффектом:")
print("   Учебное предположение: CTR/CR +10%, выручка +3% (не универсальная история")
print("   продуктов); эффект меньше MDE может стать значимым, но мощность ниже заданной.")
mde_cr_design = (Z_A + Z_B) * cr2.std(ddof=1) / cr2.mean() * math.sqrt(2 / m1["n"])
print(f"   при n дизайн-дока ({m1['n']:,}) MDE CR = {mde_cr_design:.1%} < 10% — дизайн ловит")
print(f"   p0 в дизайн-доке 23%, в синтетике вышло {cr2.mean():.1%}: p0 по пре-периоду")
print("   плывёт — а n линейно зависит от sigma^2 (урок 3.1), перепроверяйте.")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

labels = ["CTR", "CR", "CR+CUPED", "выручка", "выручка\nвинзор.", "выручка\n+CUPED"]
mde8 = [float(r["MDE отн. (n=8k)"].rstrip("%")) for r in rows]
typ = [10, 10, 10, 3, 3, 3]
bars = ax1.bar(labels, mde8,
               color=["#8172B3", "#4C72B0", "#64B5CD", "#C44E52", "#DD8452", "#937860"])
for b, v, t in zip(bars, mde8, typ):
    ax1.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                 textcoords="offset points", ha="center", fontsize=9)
    ax1.plot([b.get_x(), b.get_x() + b.get_width()], [t, t], color="k", lw=1.6)
ax1.plot([], [], color="k", lw=1.6, label="типичный эффект механики (история тестов)")
ax1.set_ylabel("MDE (отн.) при n=8 000/группу")
ax1.set_title("Чувствительность кандидатов:\nвыручка (и её усиления) не видит свои типичные эффекты 3%")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3, axis="y")

corr = [float(r["corr с выручкой"].replace(",", ".")) for r in rows]
colors6 = ["#8172B3", "#4C72B0", "#64B5CD", "#C44E52", "#DD8452", "#937860"]
ax2.scatter(corr, mde8, s=90, color=colors6)
for x, y, lab, c in zip(corr, mde8,
                        ["CTR", "CR", "CR+CUPED", "выручка", "винзор.", "+CUPED"], colors6):
    ax2.annotate(lab, (x, y), xytext=(7, 4), textcoords="offset points", fontsize=9)
ax2.set_xlabel("близость к деньгам: корреляция с выручкой")
ax2.set_ylabel("MDE (отн.), вниз = чувствительнее")
ax2.set_title("Карта OEC: вниз-вправо лучше;\nCTR — чувствительный прокси, CR — компромисс-драйвер, выручка — guardrail")
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_4_1_oec_map.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_4_1_oec_map.png")

# %%
print()
print("=" * 78)
print("ГЛАВНОЕ (что должны увидеть):")
print("1) Дизайн-док собирается функцией из гипотезы X->Y->Z + метрик + MDE:")
print("   n, даты, decision rule и ramp фиксируются ДО запуска — это контракт.")
print("2) «⅓ идей выигрывают»: без тестов квартал заканчивается около 0%;")
print("   тесты оставляют выигрыш и отсекают вредные раскатки; приоритизация")
print("   поднимает итог при той же мощности платформы (12 тестов).")
print("3) OEC: CTR чувствителен (MDE ~4%), но слабо связан с выручкой — прокси")
print("   (Гудхарт); выручка — прямые деньги, но MDE 13-17% при типичных")
print("   эффектах 3% — мощность ниже 80%, в том числе после коррекции;")
print("   CR — кандидат при доказанной связи с ценностью; деньги также можно выбрать OEC.")
