# -*- coding: utf-8 -*-
"""Практика 6.1 — Корреляция != причинность: Симпсон, коллайдер, мини-DAG.

Принцип курса: не верь — проверь симуляцией. Три блока:

1) парадокс Симпсона на продуктовых данных: платформа (iOS/Android) —
   конфаундер, который переворачивает знак эффекта новой выдачи:
   внутри обеих платформ эффект -0,5 п.п., в агрегате около +3 п.п.;
   поправка по платформе (backdoor-взвешивание) возвращает истинный эффект;

2) коллайдер: «проблемы с заказом» и «вовлечённость» НЕ зависимы, но обе
   увеличивают шанс попасть в датасет обращений в поддержку. Фильтр по
   коллайдеру рождает корреляцию из ничего (парадокс Берксона):
   corr ~ 0 на всех данных и заметно < 0 среди «попавших в датасет»;

3) мини-функция «нарисуй DAG текстом»: список рёбер + проверка открытости
   пути (d-разделение для одного пути: цепь/вилка блокируются
   обусловливанием, коллайдер наоборот открывается). Без graphviz.

Запуск:  python3 practice_6_1.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# # Практика 6.1 — Три способа обмануться корреляцией
# Кейс «ЕдаДома»: новая сортировка выдачи ресторанов, обращения в поддержку,
# и DAG «погода — заказы — доставка — выручка».

# %%
import matplotlib

matplotlib.use("Agg")
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(61)  # seed -> результат воспроизводим
HERE = Path(__file__).resolve().parent

# %% [markdown]
# ## Шаг 1. Парадокс Симпсона: платформа переворачивает эффект
#
# Легенда: «умная выдача» раскатывалась наблюдательно, без А/Б — сначала на
# iOS (у iOS-команды был релизный поезд раньше). Значит Treatment (видит новую
# выдачу) сильно скоррелирован с платформой, а платформа сама по себе
# определяет конверсию. Истинный каузальный эффект — одинаковый в обоих
# сегментах: -0,5 п.п. (выдача чуть хуже). Что покажет агрегат?

# %%
N = 200_000
P_IOS = 0.40  # доля iOS в трафике
P_TREAT = {"ios": 0.90, "android": 0.30}  # кому досталась новая выдача
CONV0 = {"ios": 0.120, "android": 0.060}  # конверсия СТАРОЙ выдачи
TRUE_EFFECT = -0.005  # истинный эффект новой выдачи: -0,5 п.п. в обеих платформах

platform = np.where(RNG.random(N) < P_IOS, "ios", "android")
treat = np.array([RNG.random() < P_TREAT[p] for p in platform]).astype(int)
p_conv = np.array([CONV0[p] for p in platform]) + TRUE_EFFECT * treat
orders = RNG.random(N) < p_conv

df = pd.DataFrame({"platform": platform, "treat": treat, "order": orders})

seg = (
    df.groupby(["platform", "treat"])["order"]
    .agg(["mean", "size"])
    .rename(columns={"mean": "конверсия", "size": "n"})
)
print("=" * 78)
print("1) ПАРАДОКС СИМПСОНА: новая выдача, наблюдательная раскатка")
print("-" * 78)
print(seg.assign(конверсия=seg["конверсия"].map("{:.2%}".format)).to_string())

naive = df.loc[df.treat == 1, "order"].mean() - df.loc[df.treat == 0, "order"].mean()
diff_ios = (
    df.query("platform == 'ios' and treat == 1")["order"].mean()
    - df.query("platform == 'ios' and treat == 0")["order"].mean()
)
diff_andr = (
    df.query("platform == 'android' and treat == 1")["order"].mean()
    - df.query("platform == 'android' and treat == 0")["order"].mean()
)
# backdoor-поправка: взвешиваем сегментные разности долями платформ (как в популяции)
w_ios = (df.platform == "ios").mean()
stratified = w_ios * diff_ios + (1 - w_ios) * diff_andr

print(f"\n   агрегат (T против C по всем)        : {naive:+.4f} = {naive:+.1%}")
print(f"   внутри iOS                          : {diff_ios:+.4f} = {diff_ios:+.1%}")
print(f"   внутри Android                      : {diff_andr:+.4f} = {diff_andr:+.1%}")
print(f"   стратифицированная оценка (по платформе): {stratified:+.4f}")
print(f"   истинный эффект (мы его задали)     : {TRUE_EFFECT:+.4f}")
print("   -> агрегат врёт знаком: +3 п.п. вместо -0,5 п.п.; поправка снимает смещение.")

# проверка «на глаз», что платформа — конфаундер: состав групп
share = df.groupby("treat")["platform"].apply(lambda s: (s == "ios").mean())
print(f"\n   доля iOS среди новой выдачи: {share[1]:.0%}, среди старой: {share[0]:.0%}")
print("   Treatment скоррелирован с платформой -> это backdoor-путь, а не эффект.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
colors = {"ios": "#4C72B0", "android": "#DD8452"}
for ax, (plat, sub) in zip(axes[:2], df.groupby("platform")):
    rates = sub.groupby("treat")["order"].mean()
    bars = ax.bar(["старая выдача", "новая выдача"], [rates[0], rates[1]],
                  color=["#C9C9C9", colors[plat]], width=0.55)
    for b, r in zip(bars, [rates[0], rates[1]]):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.002, f"{r:.2%}",
                ha="center", fontsize=10, fontweight="bold")
    d = rates[1] - rates[0]
    ax.set_title(f"{plat.upper()} (n={len(sub):,}): эффект {d:+.1%}", fontsize=11)
    ax.set_ylabel("конверсия в заказ")
    ax.set_ylim(0, 0.145)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    f"Внутри обеих платформ новая выдача ХУЖЕ ({TRUE_EFFECT:+.1%}), "
    f"но в агрегате {naive:+.1%}:\nплатформа — конфаундер (iOS раньше получили выдачу и выше конверсия)",
    fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(HERE / "practice_6_1_simpson.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_6_1_simpson.png")

# %% [markdown]
# ## Шаг 2. Коллайдер: корреляция из ничего
#
# Две НЕЗАВИСИМЫЕ причины пишут в поддержку: X = «проблемы с заказом»,
# Z = «вовлечённость» (чем больше заказываешь, тем чаще вообще пишешь).
# В датасет попадает обращение, если X + Z + шум > порога — то есть мы
# обусловливаемся на общем следствии (коллайдере). Что происходит с
# корреляцией X и Z среди «видимых в данных»?

# %%
N2 = 30_000
problems = RNG.normal(0, 1, N2)      # X: проблемы с логистикой
engagement = RNG.normal(0, 1, N2)    # Z: вовлечённость (независима от X!)
ticket_score = problems + engagement + RNG.normal(0, 0.5, N2)  # коллайдер: шанс написать
in_data = ticket_score > 0           # фильтр отбора в датасет обращений

r_all = np.corrcoef(problems, engagement)[0, 1]
r_sel = np.corrcoef(problems[in_data], engagement[in_data])[0, 1]
slope_all, intercept_all, _, p_all, _ = stats.linregress(problems, engagement)
slope_sel, intercept_sel, _, p_sel, _ = stats.linregress(problems[in_data], engagement[in_data])

print("=" * 78)
print("2) КОЛЛАЙДЕР / ПАРАДОКС БЕРКСОНА: отбор в датасет обращений")
print("-" * 78)
print(f"   истинная связь X и Z                 : 0 (задали независимыми)")
print(f"   corr на всех юзерах                  : {r_all:+.3f} (p = {p_all:.2f}) — как и должно быть")
print(f"   corr среди написавших в поддержку    : {r_sel:+.3f} "
      f"(p {'< 1e-300' if p_sel == 0 else f'= {p_sel:.1e}'})")
print(f"   наклон регрессии Z ~ X: всех {slope_all:+.3f} -> в датасете {slope_sel:+.3f}")
print(f"   доля попавших в датасет: {in_data.mean():.0%}")
print("   -> среди написавших проблемные и лояльные «меняются местами»:")
print("      у кого мало проблем — те фанаты, у кто не фанат — те с проблемами.")
print("      Выводы о связи по отфильтрованным данным — ложные.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), sharex=True, sharey=True)
idx_all = RNG.choice(N2, 4000, replace=False)
idx_sel = RNG.choice(np.flatnonzero(in_data), 4000, replace=False)
axes[0].scatter(problems[idx_all], engagement[idx_all], s=6, alpha=0.35, color="#4C72B0")
axes[0].set_title(f"Все юзеры: corr = {r_all:+.3f} — независимы")
axes[1].scatter(problems[idx_sel], engagement[idx_sel], s=6, alpha=0.35, color="#C44E52")
axes[1].set_title(f"Только написавшие в поддержку: corr = {r_sel:+.3f}")
xs = np.linspace(-4, 4, 50)
axes[1].plot(xs, intercept_sel + slope_sel * xs, "k--", lw=2,
             label=f"Z ~ X: наклон {slope_sel:+.2f}")
axes[1].legend(fontsize=9)
for ax in axes:
    ax.set_xlabel("проблемы с заказом (X)")
    ax.set_ylabel("вовлечённость (Z)")
fig.suptitle("Коллайдер «написал в поддержку»: фильтр по общему следствию рождает\n"
             "отрицательную корреляцию двух независимых причин (парадокс Берксона)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(HERE / "practice_6_1_collider.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_6_1_collider.png")

# %% [markdown]
# ## Шаг 3. Мини-DAG текстом: рёбра, пути, d-разделение
#
# DAG задаем списком рёбер. Для конкретного пути правило блокировки одно:
# смотрим на каждый промежуточный узел.
#   цепь  A -> M -> B  и вилка A <- M -> B : путь ОТКРЫТ, пока M не обусловлена;
#   коллайдер A -> C <- B                 : путь ЗАКРЫТ, пока C не обусловлена.
# Два узла d-разделены, если ЗАБЛОКИРОВАНЫ ВСЕ пути между ними.

# %%
EDGES_EDA = [
    ("weather", "orders"),         # дождь -> больше заказов (не выходят из дома)
    ("weather", "delivery_time"),  # дождь -> дольше едет курьер
    ("promo", "orders"),           # промо -> больше заказов
    ("orders", "revenue"),         # заказы -> выручка
    ("delivery_time", "revenue"),  # долгая доставка -> часть корзин не доезжает
    ("income", "promo"),           # промо таргетируется на платёжеспособные районы
    ("income", "revenue"),         # богатые районы больше тратят
]


def draw_dag_text(edges):
    """«Рисуем» DAG текстом: для каждого узла — родители и дети."""
    nodes = sorted({v for e in edges for v in e})
    lines = []
    for v in nodes:
        parents = [a for a, b in edges if b == v]
        children = [b for a, b in edges if a == v]
        lines.append(f"  {v:<14} <- {', '.join(parents) or '-':<28} -> {', '.join(children) or '-'}")
    return "\n".join(lines)


def simple_paths(edges, source, target):
    """Все простые пути source..target по скелету графа (без повторов узлов)."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    paths, stack = [], [(source, [source])]
    while stack:
        node, path = stack.pop()
        if node == target:
            paths.append(path)
            continue
        for nxt in adj.get(node, []):
            if nxt not in path:
                stack.append((nxt, path + [nxt]))
    return paths


def path_blocked(edges, path, conditioned):
    """Открыт ли конкретный путь при обусловливании на множество узлов.

    Промежуточный узел M с соседями A, B на пути:
      A -> M -> B  или  A <- M -> B  (не-коллайдер): блокируется, если M обусловлена;
      A -> M <- B  (коллайдер):      блокирован, пока M НЕ обусловлена.
    (Упрощение: потомков коллайдера не учитываем.)
    """
    es = set(edges)
    for a, m, b in zip(path, path[1:], path[2:]):
        is_collider = (a, m) in es and (b, m) in es  # обе стрелки входят в M
        if is_collider and m not in conditioned:
            return True  # коллайдер закрыт -> путь заблокирован
        if not is_collider and m in conditioned:
            return True  # цепь/вилка закрыта обусловливанием
    return False


def d_separated(edges, x, y, conditioned=(), verbose=True):
    """d-разделены ли x и y: все простые пути между ними заблокированы."""
    cond = set(conditioned)
    paths = simple_paths(edges, x, y)
    if not paths:
        return True
    open_paths = [p for p in paths if not path_blocked(edges, p, cond)]
    if verbose:
        for p in paths:
            status = "ЗАБЛОКИРОВАН" if path_blocked(edges, p, cond) else "открыт    "
            print(f"      {'-'.join(p):<44} {status}")
    return not open_paths


print("=" * 78)
print("3) МИНИ-DAG «ЕдаДома»: погода/промо/заказы/доставка/выручка/доход района")
print("-" * 78)
print(draw_dag_text(EDGES_EDA))

cases = [
    ("promo", "weather", set()),           # коллайдер `orders` закрыт по умолчанию
    ("promo", "weather", {"orders"}),      # ...и открывается, стоит обусловить orders
    ("promo", "revenue", {"orders"}),      # «causal salad»: заодно открыли коллайдер
    ("promo", "revenue", {"income"}),      # backdoor-путь закрыт, каузальный живёт
]
for x, y, cond in cases:
    print(f"\n   d-разделены ли {x} и {y} при обусловленных {cond or '{}'}?")
    sep = d_separated(EDGES_EDA, x, y, cond)
    print(f"   ОТВЕТ: {'ДА (связи нет)' if sep else 'НЕТ (путь открыт)'}")

print("\n   Читаем: promo и погода независимы, ПОКА мы не законтролировали orders")
print("   (коллайдер открылся). А «регрессия выручки на промо + orders + ещё 30")
print("   ковариат» открывает именно такие пути — causal salad из урока.")

# мини-автотест логики на трёх базовых структурах
chain = [("a", "m"), ("m", "b")]
fork = [("z", "a"), ("z", "b")]
coll = [("a", "c"), ("b", "c")]
assert d_separated(chain, "a", "b", set(), verbose=False) is False
assert d_separated(chain, "a", "b", {"m"}, verbose=False) is True
assert d_separated(fork, "a", "b", {"z"}, verbose=False) is True
assert d_separated(coll, "a", "b", set(), verbose=False) is True
assert d_separated(coll, "a", "b", {"c"}, verbose=False) is False
print("\n   [автотест] цепь/вилка/коллайдер ведут себя ровно как в теории: OK")
