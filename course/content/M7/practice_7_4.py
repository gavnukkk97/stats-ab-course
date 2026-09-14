# -*- coding: utf-8 -*-
"""Практика 7.4 — Платформа как календарь гипотез
(сквозной кейс «ЕдаДома», урок 7.4 модуля «Проектирование АБ-платформы»).

Принцип курса: не верь интуиции — проверь симуляцией.

Идея урока (видение заказчика): слоты трафика — конечный ресурс, как
переговорные. Календарь = решётка «слой × неделя»; система сама считает
длительность теста из MDE (урок 3.1) и предлагает ближайший свободный слот,
а очередь приоритизируется по ЭКОНОМИКЕ СЛОТА, а не по ICE-баллам.

Что делаем:
1) Бэклог из 15 гипотез «ЕдаДома»: для каждой MDE -> n -> длительность
   (формула урока 3.1), слой (конфликты), ценность (млн ₽/нед на раскатке)
   и вероятность успеха (зелёный тест);
2) жадный планировщик в решётку «слои × недели» с приоритетом
   «ценность на слот-неделю» = p_успеха * ценность / длительность
   (простая версия E[Δ^+] Аzevedo: ценность выше у неопределённых, но
   потенциально крупных гипотез);
3) сравнение с FIFO «кто первый»: реализованная ценность квартала,
   среднее время ожидания, throughput;
4) чувствительность: трафик ×2 — появляется ли свободная ёмкость;
5) сезонность: blackout-недели пиковых продаж — старты запрещены;
6) паспорт ёмкости платформы: сколько тестов/квартал она тянет.

Запуск:  python3 practice_7_4.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# Шаг 0. Импорты и константы: трафик 11 700 юзеров/нед на обе группы (3.1),
# квартал = 13 недель, 3 слоя (домена изменений, урок 4.2), blackout-недели
# 10–11 — пиковые продажи («Чёрная пятница»): новые тесты не стартуют,
# чтобы не мешать продаже и не ловить сезонные артефакты в данные.

# %%
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260914)
HERE = Path(__file__).resolve().parent

ALPHA, POWER = 0.05, 0.80
Z1 = stats.norm.ppf(1 - ALPHA / 2)   # 1.96
Z2 = stats.norm.ppf(POWER)           # 0.84
WEEKLY = 11_700                     # юзеров/нед на обе группы
QUARTER = 13                        # недель в квартале
BLACKOUT = set(range(10, 12))       # недели 10–11: старты запрещены
LAYERS = ["ранжирование", "интерфейс", "коммуникации"]


def n_per_group(p0, mde_rel):
    """n на группу для бинарной OEC (формула урока 3.1, запас 5%)."""
    sigma = math.sqrt(p0 * (1 - p0))
    return math.ceil(1.05 * 2 * (Z1 + Z2) ** 2 * sigma ** 2 / (mde_rel * p0) ** 2)


def duration_weeks(h, weekly):
    """Длительность теста целыми неделями (минимум 2 — недельная сезонность)."""
    return max(2, math.ceil(2 * h["n"] / weekly))


# %% [markdown]
# Шаг 1. Бэклог квартала: 15 гипотез. Приоритет «кто первый» = порядок
# поступления (H01..H15). Экономика слота: score = p_успеха * ценность /
# длительность — «сколько млн ожидаемой ценности в неделю занятия слота».
# Вероятности успеха калиброваны под «⅓ идей выигрывают» (Kohavi KDD'13).

# %%
@dataclass
class Hyp:
    hid: str
    name: str
    layer: str
    p0: float        # базовая конверсия OEC
    mde_rel: float   # MDE, % отн. (из экономики, уроки 3.1/4.5)
    value: float     # ценность при раскатке, млн ₽/нед
    p_succ: float    # вероятность зелёного теста (⅓ выигрывают)


BACKLOG_RAW = [
    # ранжирование
    Hyp("H01", "Умная сортировка по ETA", "ранжирование", 0.23, 0.065, 2.8, 0.45),
    Hyp("H04", "Рейтинг в ранжировании", "ранжирование", 0.23, 0.050, 1.2, 0.30),
    Hyp("H07", "Бейдж «быстрее 30 минут»", "ранжирование", 0.12, 0.083, 3.2, 0.25),
    Hyp("H10", "Бонусы курьерам в рейтинге", "ранжирование", 0.45, 0.050, 0.8, 0.20),
    Hyp("H13", "«Любимые» рестораны наверху", "ранжирование", 0.23, 0.040, 1.9, 0.35),
    # интерфейс
    Hyp("H02", "Новые иконки разделов", "интерфейс", 0.23, 0.040, 0.5, 0.25),
    Hyp("H05", "One-click reorder", "интерфейс", 0.23, 0.065, 2.2, 0.40),
    Hyp("H08", "Чекаут: 3 шага -> 1", "интерфейс", 0.23, 0.050, 3.0, 0.35),
    Hyp("H11", "Тёмная тема", "интерфейс", 0.45, 0.050, 0.4, 0.15),
    Hyp("H14", "Карта доставки в карточке", "интерфейс", 0.12, 0.100, 1.1, 0.30),
    # коммуникации
    Hyp("H03", "Пуш о брошенной корзине", "коммуникации", 0.062, 0.16, 1.5, 0.55),
    Hyp("H06", "Статус заказа в мессенджере", "коммуникации", 0.062, 0.25, 0.9, 0.50),
    Hyp("H09", "Email-дайджест меню", "коммуникации", 0.050, 0.20, 0.7, 0.35),
    Hyp("H12", "Пуш: скидка 100 ₽ новичку", "коммуникации", 0.062, 0.16, 2.6, 0.40),
    Hyp("H15", "Реферальная программа", "коммуникации", 0.030, 0.30, 1.3, 0.30),
]

# расчёт n и длительностей при базовом и удвоенном трафике
BACKLOG = []
for h in BACKLOG_RAW:
    d = dict(h.__dict__)
    d["n"] = n_per_group(d["p0"], d["mde_rel"])
    d["dur_base"] = duration_weeks(d, WEEKLY)
    d["dur_x2"] = duration_weeks(d, 2 * WEEKLY)
    d["score"] = d["p_succ"] * d["value"] / d["dur_base"]
    BACKLOG.append(d)
df_backlog = pd.DataFrame(BACKLOG)[["hid", "name", "layer", "p0", "mde_rel",
                                     "n", "dur_base", "dur_x2", "value",
                                     "p_succ", "score"]]

# один и тот же «мир» для всех стратегий: зелёность гипотезы фиксирована
GREEN = {h["hid"]: bool(RNG.random() < h["p_succ"]) for h in BACKLOG}

print("=" * 78)
print("1) БЭКЛОГ КВАРТАЛА: 15 ГИПОТЕЗ, MDE -> n -> ДЛИТЕЛЬНОСТЬ (урок 3.1)")
print("-" * 78)
with pd.option_context("display.width", 110):
    print(df_backlog.assign(
        mde_rel=lambda d: (d["mde_rel"] * 100).round(1).astype(str) + "%",
        p_succ=lambda d: d["p_succ"].round(2),
        score=lambda d: d["score"].round(3),
        n=lambda d: d["n"].map("{:,}".format),
    ).to_string(index=False))
demand = df_backlog.groupby("layer")["dur_base"].sum()
print(f"\nспрос на слоты по слоям (слот-недель): "
      f"{demand.to_dict()} | всего {demand.sum()} при ёмкости "
      f"{len(LAYERS) * QUARTER} -> очередь неизбежна")
print(f"зелёность в этом «мире»: {sum(GREEN.values())} из {len(GREEN)} гипотез "
      f"(ожидание ~1/3, Kohavi KDD'13)")

# %% [markdown]
# Шаг 2. Планировщик. Правила:
# - слой занят одним тестом (доля 100%); старты в blackout-недели запрещены;
# - тест запускается, только если успевает закончиться до конца квартала;
# - FIFO: внутри слоя порядок «кто первый принёс»;
# - SLOT-экономика: первой идёт гипотеза с максимальным score
#   = p_успеха * ценность / длительность (млн ₽ ожидания за слот-неделю).

# %%
def plan(backlog, rule, weekly):
    """Расписание квартала. Возвращает список слотов и метрики."""
    durs = {h["hid"]: (h["dur_base"] if weekly == WEEKLY else h["dur_x2"])
            for h in backlog}
    slots = []
    for layer in LAYERS:
        queue = [h for h in backlog if h["layer"] == layer]
        if rule == "SLOT":                      # экономика слота
            queue = sorted(queue, key=lambda h: -h["score"])
        t = 0
        for h in queue:
            start = t
            while start in BLACKOUT:            # сезонность: сдвигаем старт
                start += 1
            dur = durs[h["hid"]]
            if start + dur > QUARTER:           # не успевает -> следующий квартал
                slots.append(dict(hid=h["hid"], layer=layer, start=np.nan,
                                  finish=np.nan, dur=dur, tested=False))
                continue
            finish = start + dur
            tested = True
            green = GREEN[h["hid"]]
            value = h["value"] * (QUARTER - finish) if green else 0.0
            slots.append(dict(hid=h["hid"], layer=layer, start=start,
                              finish=finish, dur=dur, tested=tested,
                              green=green, value=value))
            t = finish
    df = pd.DataFrame(slots)
    tested = df[df["tested"]]
    metrics = dict(
        rule=rule, weekly=weekly,
        tested=int(df["tested"].sum()),
        value=float(tested["value"].sum()),
        wait_mean=float(tested["start"].mean()),       # все заявки пришли в нед. 0
        decision_mean=float(tested["finish"].mean()),  # идея -> решение
        throughput=float(df["tested"].sum() / QUARTER),
    )
    return df, metrics


df_fifo, m_fifo = plan(BACKLOG, "FIFO", WEEKLY)
df_slot, m_slot = plan(BACKLOG, "SLOT", WEEKLY)
df_fifo2, m_fifo2 = plan(BACKLOG, "FIFO", 2 * WEEKLY)
df_slot2, m_slot2 = plan(BACKLOG, "SLOT", 2 * WEEKLY)

rows = [m_fifo, m_slot, m_fifo2, m_slot2]
labels = ["FIFO «кто первый»", "экономика слота",
          "FIFO, трафик ×2", "экономика слота, трафик ×2"]
print()
print("=" * 78)
print("2) КВАРТАЛ: FIFO «КТО ПЕРВЫЙ» vs ЭКОНОМИКА СЛОТА (+ ЧУВСТВИТЕЛЬНОСТЬ)")
print("-" * 78)
print(f"   {'стратегия':<30} | {'тестов':>6} | {'ценность, млн ₽':>15} | "
      f"{'ср. ожидание, нед':>17} | {'решение, нед':>12} | {'тест/нед':>8}")
print(f"   {'-'*30}-+-{'-'*6}-+-{'-'*15}-+-{'-'*17}-+-{'-'*12}-+-{'-'*8}")
for r, lab in zip(rows, labels):
    print(f"   {lab:<30} | {r['tested']:>6} | {r['value']:>15.1f} | "
          f"{r['wait_mean']:>17.1f} | {r['decision_mean']:>12.1f} | "
          f"{r['throughput']:>8.2f}")
print("\n   ценность = Σ по зелёным тестам: ценность/нед × недели раскатки "
      "до конца квартала;")
print("   «мир» один и тот же (зелёность фиксирована) — разница только "
      "в расписании.")

# %% [markdown]
# Шаг 3. «Принёс гипотезу — получил ближайший слот»: что отвечает календарь
# гипотезам, не поместившимся в квартал (демо для самой дорогой из очереди).

# %%
print()
print("=" * 78)
print("3) «ПРИНЁС ГИПОТЕЗУ -> ПОЛУЧИЛ СЛОТ»: ОТВЕТЫ КАЛЕНДАРЯ (базовый трафик)")
print("-" * 78)
for rule, df in (("FIFO", df_fifo), ("SLOT", df_slot)):
    left = df[~df["tested"]]
    for _, s in left.iterrows():
        h = next(b for b in BACKLOG if b["hid"] == s["hid"])
        layer_free = int(df[(df["layer"] == s["layer"]) & df["tested"]]["finish"].max())
        start = layer_free
        while start in BLACKOUT:
            start += 1
        print(f"[{rule}] {s['hid']} «{h['name']}»: слой «{s['layer']}» свободен "
              f"с нед. {start} + длительность {s['dur']} нед -> финиш "
              f"{start + s['dur']} > {QUARTER}; ближайший слот — Q1-2027; "
              f"score = {h['score']:.2f} млн/слот-нед")

# %% [markdown]
# Шаг 4. Паспорт ёмкости: сколько тестов тянет платформа. Ёмкость =
# слоистость × недели / средняя длительность; длительность — функция MDE-микса
# бэклога и трафика (статистическая ёмкость; вычислительная — отдельный слой,
# урок 7.2).

# %%
print()
print("=" * 78)
print("4) ПАСПОРТ ЁМКОСТИ ПЛАТФОРМЫ (статистическая ёмкость)")
print("-" * 78)
for tag, mult in (("база", 1), ("трафик ×2", 2)):
    dur_key = "dur_base" if mult == 1 else "dur_x2"
    mean_dur = df_backlog[dur_key].mean()
    capacity = len(LAYERS) * QUARTER / mean_dur
    demand_wk = df_backlog[dur_key].sum()
    print(f"   {tag:<10}: средняя длительность {mean_dur:.1f} нед -> ёмкость "
          f"~{capacity:.1f} тестов/квартал (3 слоя × 13 нед / {mean_dur:.1f}); "
          f"спрос бэклога {demand_wk:.0f} слот-недель из {len(LAYERS)*QUARTER}")
slack_base = len(LAYERS) * QUARTER - df_backlog["dur_base"].sum()
slack_x2 = len(LAYERS) * QUARTER - df_backlog["dur_x2"].sum()
print(f"\n   свободная ёмкость: база {slack_base:+.0f} слот-недель (платформа — "
      f"узкое место), трафик ×2: {slack_x2:+.0f} (появился резерв: {m_fifo2['tested']}"
      f"/15 гипотез протестировано даже по FIFO)")
print("   вывод: узкое место смещается — платформа перестаёт лимитировать,")
print("   начинает лимитировать генерация гипотез и скорость команд (4.5).")

# %% [markdown]
# Шаг 5. Визуализация: календарь-гант «слои × недели», FIFO vs экономика
# слота; blackout-недели подсвечены; зелёный = раскатка, серый = тест без
# эффекта, красный крест = не поместилось (следующий квартал).

# %%
fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6), sharey=True)
for ax, (df, m, title) in zip(axes, (
        (df_fifo, m_fifo, f"FIFO «кто первый»: {m_fifo['tested']} тестов, "
                          f"{m_fifo['value']:.1f} млн ₽"),
        (df_slot, m_slot, f"экономика слота: {m_slot['tested']} тестов, "
                          f"{m_slot['value']:.1f} млн ₽"))):
    ax.axvspan(10, 12, color="#DD8452", alpha=0.15)
    ax.text(11, 2.55, "blackout:\nпиковые продажи", ha="center", fontsize=8,
            color="#B2611E")
    for _, s in df.iterrows():
        y = LAYERS.index(s["layer"])
        if s["tested"]:
            color = "#55A868" if s["green"] else "#B0B0B0"
            ax.barh(y, s["dur"], left=s["start"], height=0.62, color=color,
                    edgecolor="white")
            ax.text(s["start"] + s["dur"] / 2, y, s["hid"], ha="center",
                    va="center", fontsize=8,
                    color="white" if s["green"] else "#333333")
        else:
            ax.scatter(QUARTER + 0.4, y, marker="x", s=70, color="#C44E52")
            ax.text(QUARTER + 0.7, y, s["hid"], va="center", fontsize=8,
                    color="#C44E52")
    ax.axvline(QUARTER, color="#C44E52", lw=1.4, ls=":")
    ax.set_xlim(0, QUARTER + 2.2)
    ax.set_xticks(range(0, QUARTER + 1, 2))
    ax.set_xlabel("неделя квартала")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, axis="x")
axes[0].set_yticks(range(len(LAYERS)))
axes[0].set_yticklabels(LAYERS)
fig.suptitle("Календарь гипотез «ЕдаДома», Q4-2026: решётка «слой × неделя»; "
             "зелёный — тест зелёный (раскатка), серый — нет эффекта, "
             "красный × — не поместилось", y=1.04)
fig.tight_layout()
fig.savefig(HERE / "practice_7_4_calendar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Сохранено: practice_7_4_calendar.png")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.2))
x = np.arange(len(rows))
colors = ["#4C72B0", "#55A868", "#8172B3", "#64B5CD"]
ax1.bar(x, [r["value"] for r in rows], color=colors)
for xi, r in zip(x, rows):
    ax1.annotate(f"{r['value']:.0f}", (xi, r["value"]), xytext=(0, 4),
                 textcoords="offset points", ha="center", fontsize=10)
    ax1.annotate(f"{r['tested']} тестов", (xi, 1.5), ha="center", fontsize=8,
                 color="white" if r["value"] > 25 else "#333333")
ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax1.set_ylabel("реализованная ценность квартала, млн ₽")
ax1.set_title("Экономика слота бьёт FIFO при том же трафике;\nтрафик ×2 бьёт обоих (но упирается в генерацию гипотез)")
ax1.grid(alpha=0.3, axis="y")

ax2.bar(x - 0.18, [r["wait_mean"] for r in rows], width=0.36, color="#DD8452",
        label="ср. ожидание в очереди, нед")
ax2.bar(x + 0.18, [r["decision_mean"] for r in rows], width=0.36,
        color="#4C72B0", label="ср. время идея -> решение, нед")
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax2.set_ylabel("недели")
ax2.set_title("Velocity-метрики: ожидание в очереди — управляемая часть\nцикла «идея -> решение»")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(HERE / "practice_7_4_velocity.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Сохранено: practice_7_4_velocity.png")

# %%
print()
print("=" * 78)
print("ГЛАВНОЕ (что должны увидеть):")
print("1) Спрос на слоты (46 слот-недель) > ёмкость (39) — очередь неизбежна;")
print("   вопрос только в том, КТО в ней стоит: FIFO «кто первый» или экономика")
print("   слота (p * ценность / длительность).")
print("2) Экономика слота даёт больше ценности на том же трафике тем же числом")
print("   тестов — меняется СОСТАВ очереди: вперёд идут плотные по ценности")
print("3) Трафик ×2 снимает очередь (15/15 даже по FIFO) — но тогда узкое место")
print("   смещается в генерацию гипотез: платформа перестаёт лимитировать.")
print("4) Паспорт ёмкости считается из трафика и MDE-микса бэклога — это")
print("   статистическая ёмкость, главный ограничитель платформы.")
print("5) Календарь = слой × время с учётом blackout-недель; «принёс гипотезу —")
print("   получил ближайший слот» — этого нет ни у одного вендора (white space).")
