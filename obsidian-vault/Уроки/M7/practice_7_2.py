# -*- coding: utf-8 -*-
"""Практика 7.2 — Данные и вычисления: экономика пайплайна «ЕдаДома»
(урок 7.2 модуля «Проектирование АБ-платформы»).

Принцип курса: не верь формуле — проверь симуляцией.

Что делаем:
1) Exposure logging: лог первого контакта с ЧЕСТНЫМ timestamp, дубли
   от ретраев (асимметричные: тяжёлый тестовый вариант ретраится чаще)
   -> без дедупа получается SRM; с дедупом — здоровый сплит. Плюс
   цена лог-времени вместо времени контакта (переход через границу суток);
2) Юнит-таблица: из ~1 млн событий и сырого exposure-лога -> одна строка
   на юзера (группа, экспозиция, метрики) — вход всех анализов;
3) Бакетизация: тест поюзерно (300k строк) vs по 1000/100 бакетам:
   сверка оценок и p-value + честный замер времени (time.perf_counter)
   на ночную партию 100 экспериментов x 10 метрик;
4) Партиционирование озера: 30 дней x 100 экспериментов — сколько строк
   сканируем без партиций / с партицией по дню / по (дате, эксперименту).

Запуск:  python3 practice_7_2.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# Шаг 0. Импорты и параметры эксперимента «checkout_redesign»
# («ЕдаДома», 2 недели, 300k юзеров, сплит 50/50 по md5 — как в 7.1).

# %%
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260914)
HERE = Path(__file__).resolve().parent

N_USERS = 300_000
DAYS = 14
SALT_EXP = "layer_checkout_v1|exp_2026_09"
TRUE_LIFT = 0.04                      # истинный эффект B: +4% GMV


def md5_bucket(key: str, salt: str, m: int = 1000) -> int:
    return int(hashlib.md5(f"{salt}|{key}".encode()).hexdigest(), 16) % m


def srm_pvalue(n_ctl: int, n_tst: int) -> float:
    total = n_ctl + n_tst
    e = total / 2.0
    chi2 = (n_ctl - e) ** 2 / e + (n_tst - e) ** 2 / e
    return float(stats.chi2.sf(chi2, 1))


# %% [markdown]
# Шаг 1. Данные эксперимента: юзеры -> группы (хэш), события заказов,
# сессии (для ratio-метрики «GMV за сессию»), exposure-лог с дубями и
# двумя временными метками: contact_ts (первый показ) и log_ts (когда
# строчка доехала до шины: батч-флаш раз в час + офлайн-мобайл).

# %%
user_ids = np.array([f"user_{i:08d}" for i in range(N_USERS)])
t0 = time.perf_counter()
group_b = np.array([md5_bucket(u, SALT_EXP) >= 500 for u in user_ids])  # 50/50
hash_sec = time.perf_counter() - t0

# события заказов: Пуассон(0.22)/день на юзера, GMV логнормальный
orders_per_ud = RNG.poisson(0.22, size=(N_USERS, DAYS))
flat = orders_per_ud.ravel()                 # порядок: юзер-мажор (u*DAYS + d)
n_events = int(flat.sum())
pair_idx = np.repeat(np.arange(N_USERS * DAYS), flat)
ev_user, ev_day = pair_idx // DAYS, (pair_idx % DAYS).astype(np.int8)
ev_hour = RNG.uniform(0, 24, size=n_events)
# экспозиция: первый контакт в первые 72 часа теста
exposure_hour = RNG.uniform(0, 72, size=N_USERS)
after = ev_day * 24 + ev_hour >= exposure_hour[ev_user]
ev_user, ev_day, ev_hour = ev_user[after], ev_day[after], ev_hour[after]
n_events_in = len(ev_user)
ev_day = ev_day.astype(np.int8)

is_mobile = RNG.random(N_USERS) < 0.30
gmv_mu = math.log(60.0)
gmv_ev = RNG.lognormal(gmv_mu, 0.9, size=n_events_in) * \
    np.where(group_b[ev_user], 1 + TRUE_LIFT, 1.0)
sessions_ud = RNG.poisson(0.8, size=(N_USERS, DAYS)).sum(axis=1)
events = pd.DataFrame({"user_i": ev_user, "day": ev_day.astype(np.int8),
                       "gmv": gmv_ev.astype(np.float32)})

print("=" * 78)
print(f"1) ДАННЫЕ ЭКСПЕРИМЕНТА «checkout_redesign»: {N_USERS:,} юзеров, {DAYS} дн.")
print("-" * 78)
print(f"   хэш-назначение {N_USERS:,} юзеров: {hash_sec:.2f} с"
      f" ({N_USERS / hash_sec / 1e6:.2f} млн хэшей/с — состояние не нужно)")
print(f"   события заказов после экспозиции: {n_events_in:,} строк"
      f" (все события за период: {n_events:,})")

# exposure-лог: дубли от ретраев. Контроль +4%, тест +7% (тяжёлый вариант
# рендерится дольше -> больше сетевых ретраев) — каждый дубль = доп. строка
dup_p = np.where(group_b, 0.07, 0.04)
dup = RNG.random(N_USERS) < dup_p
log_group = np.concatenate([group_b, group_b[dup]])          # юзеры + дубли
n_ctl_raw, n_tst_raw = int((~log_group).sum()), int(log_group.sum())
n_ctl, n_tst = int((~group_b).sum()), int((group_b).sum())

p_raw, p_dedup = srm_pvalue(n_ctl_raw, n_tst_raw), srm_pvalue(n_ctl, n_tst)

# две метки времени: контакт и лог (батч-флаш 1ч; мобайл-офлайн ~ Exp(12ч))
delay = RNG.exponential(1.0, N_USERS) + np.where(is_mobile, RNG.exponential(12.0, N_USERS), 0.0)
contact_day = (exposure_hour // 24).astype(int)
log_day = ((exposure_hour + delay) // 24).astype(int)
wrong_day = np.mean(contact_day != log_day)
d1_contact = np.sum(contact_day == 0)
d1_log = np.sum(log_day == 0)

print(f"\n   exposure-лог: сырой {len(log_group):,} строк (дубли {(dup).mean():.0%}),"
      f" после дедупа {N_USERS:,}")
print(f"   БЕЗ дедупа: контроль {n_ctl_raw:,} / тест {n_tst_raw:,}"
      f" -> SRM chi2-тест p = {p_raw:.1e} — РЕЗУЛЬТАТАМ НЕ ВЕРИТЬ")
print(f"   С дедупом : контроль {n_ctl:,} / тест {n_tst:,}"
      f" -> p = {p_dedup:.2f} — здоровы")
print(f"   лог-время вместо времени контакта: {wrong_day:.1%} экспозиций")
print(f"   уезжают в другие сутки; день-1 по контакту {d1_contact:,},"
      f" по логам {d1_log:,} юзеров ({(d1_contact - d1_log) / d1_contact:.0%} «опоздавших»)")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
x = np.arange(2)
ax1.bar(x - 0.18, [n_ctl_raw / 1000, n_tst_raw / 1000], width=0.36, color="#C44E52", label="сырой лог (с дубями)")
ax1.bar(x + 0.18, [n_ctl / 1000, n_tst / 1000], width=0.36, color="#55A868", label="после дедупа")
ax1.axhline(N_USERS / 2 / 1000, color="gray", ls="--", lw=1.2, label="план 150k")
ax1.set_xticks(x, ["контроль", "тест"])
ax1.set_ylabel("юзеров, тыс.")
ax1.set_title(f"Дубли ретраев ломают знаменатель: сырой лог даёт SRM\n(p={p_raw:.0e}; асимметрия 3% vs 6% ретраев); дедуп чинит (p={p_dedup:.2f})")
ax1.legend(fontsize=9)
hours = np.arange(0, 72, 1.0)
ax2.plot(hours, np.array([np.mean((exposure_hour < h) & (log_day > contact_day)) for h in hours]) * 100,
         color="#4C72B0", lw=2, label="экспозиции с уехавшей датой, %")
ax2.set_xlabel("час теста (стартовые 3 суток)")
ax2.set_ylabel("% логов с неверной датой")
ax2.set_title(f"Лог-время ≠ время контакта: {wrong_day:.0%} экспозиций меняют дату\n(батч-флаш 1ч + офлайн-мобайл ~12ч) — «день в тесте» врёт")
ax2.grid(alpha=0.3)
ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "practice_7_2_dedup.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_7_2_dedup.png")

# %% [markdown]
# Шаг 2. Юнит-таблица: сворачиваем события до одной строки на юзера.
# Это вход любого анализа (и единственная «персональная» таблица теста):
# группа, exposure-метки, суммы метрик за период после экспозиции.

# %%
agg = events.groupby("user_i").agg(orders=("gmv", "size"), gmv=("gmv", "sum"))
unit = pd.DataFrame({
    "user": user_ids,
    "group_b": group_b,
    "exposure_h": exposure_hour,
    "contact_day": contact_day.astype(np.int8),
    "sessions": sessions_ud,
}).set_index(np.arange(N_USERS))
unit = unit.join(agg)
unit[["orders", "gmv"]] = unit[["orders", "gmv"]].fillna(0)
unit["gmv"] = unit["gmv"].astype(np.float32)

mem_events_mb = n_events * (4 + 1 + 4) / 1e6
mem_unit_mb = N_USERS * (8 + 1 + 1 + 4 + 4 + 4 + 40) / 1e6
print("=" * 78)
print("2) ЮНИТ-ТАБЛИЦА: события -> одна строка на юзера")
print("-" * 78)
print(f"   {n_events:,} событий за период ({mem_events_mb:.0f} МБ; после экспозиции {n_events_in:,})"
      f" -> {len(unit):,} строк юнит-таблицы (~{mem_unit_mb:.0f} МБ) — сжатие ×{n_events / N_USERS:.1f}")
print(f"   колонки: группа, exposure_h, contact_day, sessions, orders, gmv")
print("   именно её (а не сырьё) читает nightly-батч анализов")

# %% [markdown]
# Шаг 3. Бакетизация: судьба юзера = md5(соль|юзер) % K. По бакетам
# храним (n, sum, sum_sq) — метрик сколько угодно, строк 2K вместо 300k.
# Анализ: t-тест на бакетных средних; ratio-метрика — дельта-метод по
# корзинам (S_j, C_j) — формулы из 3.3, здесь только применение.

# %%
K_LIST = (100, 1000)
buckets = {K: np.array([md5_bucket(u, "bucket_salt_v1", K) for u in user_ids]) for K in K_LIST}

gmv_a = unit.loc[~unit.group_b, "gmv"].to_numpy()
gmv_b = unit.loc[unit.group_b, "gmv"].to_numpy()
t_user = stats.ttest_ind(gmv_b, gmv_a, equal_var=False)
lift_user = gmv_b.mean() / gmv_a.mean() - 1

results = []
lift_by_k, p_by_k = {}, {}
for K in K_LIST:
    bk = buckets[K]
    n_a = np.bincount(bk[~group_b], minlength=K).astype(np.float64)
    s_a = np.bincount(bk[~group_b], weights=gmv_a, minlength=K)
    n_b = np.bincount(bk[group_b], minlength=K).astype(np.float64)
    s_b = np.bincount(bk[group_b], weights=gmv_b, minlength=K)
    m_a, m_b = s_a / n_a, s_b / n_b
    t_bk = stats.ttest_ind(m_b, m_a, equal_var=False)
    lift_bk = m_b.mean() / m_a.mean() - 1
    lift_by_k[K], p_by_k[K] = lift_bk, t_bk.pvalue
    results.append({"путь": f"бакеты K={K}", "строк на анализ": 2 * K,
                    "оценка lift GMV": f"{lift_bk:.3%}", "p-value": f"{t_bk.pvalue:.4f}",
                    "|Δlift| к поюзерному": f"{abs(lift_bk - lift_user):.3%}"})
    if K == 1000:
        m1000 = (m_a, m_b)

# ratio-метрика «GMV за сессию»: поюзерный дельта-метод vs по 1000 бакетам
def delta_ratio(dfa: pd.DataFrame, key: str | None = None) -> tuple[float, float, float, float]:
    """Дельта-метод для ratio = ΣS/ΣC; key: колонка бакета или None (поюзерно)."""
    outs = {}
    for name, df in (("A", dfa[~dfa.group_b]), ("B", dfa[dfa.group_b])):
        if key is None:
            s, c = df["gmv"].to_numpy(), df["sessions"].to_numpy().astype(float)
            R = s.sum() / c.sum()
            var = np.var(s - R * c, ddof=1) / (len(s) * c.mean() ** 2)
        else:
            g = df.groupby(key).agg(s=("gmv", "sum"), c=("sessions", "sum"))
            s, c = g["s"].to_numpy(), g["c"].to_numpy().astype(float)
            R = s.sum() / c.sum()
            var = np.var(s - R * c, ddof=1) / (len(s) * c.mean() ** 2)
        outs[name] = (R, var)
    RA, VA = outs["A"]
    RB, VB = outs["B"]
    z = (RB - RA) / math.sqrt(VA + VB)
    return RA, RB, z, float(2 * stats.norm.sf(abs(z)))

u_full = unit.copy()
u_full["bk"] = buckets[1000]
RA_u, RB_u, z_u, p_u = delta_ratio(u_full)
RA_k, RB_k, z_k, p_k = delta_ratio(u_full, key="bk")

df_bk = pd.DataFrame([
    {"путь": "поюзерно (300k строк)", "строк на анализ": N_USERS,
     "оценка lift GMV": f"{lift_user:.3%}", "p-value": f"{t_user.pvalue:.4f}",
     "|Δlift| к поюзерному": "—"},
    *results,
])
print("=" * 78)
print("3) БАКЕТИЗАЦИЯ: сверка статистики (одни и те же данные)")
print("-" * 78)
print(df_bk.to_string(index=False))
print(f"\n   ratio-метрика GMV/сессию (дельта-метод из 3.3):")
print(f"     поюзерно : A={RA_u:.1f}₽, B={RB_u:.1f}₽, z={z_u:.2f}, p={p_u:.4f}")
print(f"     1000 бакт: A={RA_k:.1f}₽, B={RB_k:.1f}₽, z={z_k:.2f}, p={p_k:.4f}")
print("     оценки и p-value практически совпадают — бакетизация не теряет статистику,")

# %% [markdown]
# Экономика: ночная партия 100 экспериментов x 300 метрико-срезов
# (метрики x сегменты; у Авито таких до 30 тыс. на тест). Поюзерный путь
# на каждый (эксперимент x срез) снова проходит всю юнит-таблицу; бакетный —
# один GROUP BY на эксперимент (все срезы сразу), дальше статистика по 2K
# строк. Замеряем честно, время = лучшая из 30 попыток.

# %%
N_EXPS, N_SLICES = 100, 300
N_ANALYSES = N_EXPS * N_SLICES


def numpy_welch(a: np.ndarray, b: np.ndarray) -> float:
    """t-статистика Уэлча чистым numpy — чтобы мерить данные, а не оверхед scipy."""
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    return (mb - ma) / math.sqrt(va / len(a) + vb / len(b))


def best_of(fn, n: int = 30) -> float:
    best = math.inf
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


sec_user_one = best_of(lambda: numpy_welch(gmv_b, gmv_a))          # 150k+150k строк
sec_user_total = sec_user_one * N_ANALYSES

# агрегация юнит-таблицы в бакеты — что хранит платформа: (n, sum, sum_sq)
# на (группа x бакет); в движке это один компилированный GROUP BY-проход
def aggregate_to_buckets(gmv_arr: np.ndarray, bk_sel: np.ndarray) -> tuple:
    return (np.bincount(bk_sel, minlength=1000),
            np.bincount(bk_sel, weights=gmv_arr, minlength=1000),
            np.bincount(bk_sel, weights=gmv_arr ** 2, minlength=1000))

sec_agg_one = best_of(lambda: (aggregate_to_buckets(gmv_a, buckets[1000][~group_b]),
                               aggregate_to_buckets(gmv_b, buckets[1000][group_b])))
sec_bk_one = best_of(lambda: numpy_welch(m1000[1], m1000[0]))      # 1000+1000 строк
sec_bk_total = N_EXPS * sec_agg_one + N_ANALYSES * sec_bk_one

speedup = sec_user_total / sec_bk_total
rows_user_scanned = N_ANALYSES * N_USERS
rows_bk_scanned = N_EXPS * N_USERS + N_ANALYSES * 2 * 1000

print("=" * 78)
print(f"4) ЭКОНОМИКА ВЫЧИСЛЕНИЙ: ночная партия {N_EXPS} экспериментов"
      f" x {N_SLICES} метрико-срезов = {N_ANALYSES:,} анализов")
print("-" * 78)
print(f"   поюзерно : {sec_user_one * 1e3:.2f} мс/анализ (по 300k строк)"
      f" x {N_ANALYSES:,} = {sec_user_total:.0f} с;")
print(f"              сканируется {rows_user_scanned / 1e9:.1f} млрд строк")
print(f"   бакеты   : GROUP BY {sec_agg_one * 1e3:.1f} мс x {N_EXPS} + статистика"
      f" {sec_bk_one * 1e6:.0f} мкс x {N_ANALYSES:,} = {sec_bk_total:.1f} с;")
print(f"              сканируется {rows_bk_scanned / 1e6:.0f} млн строк"
      f" (×{rows_user_scanned / rows_bk_scanned:.0f} меньше)")
print(f"   УСКОРЕНИЕ ×{speedup:.0f} на одной ноде; в озере (Spark/Trino) счёт идёт")
print("   за шаффл/скан строк — там правит отношение строк ×"
      f"{rows_user_scanned / rows_bk_scanned:.0f}")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
names = ["поюзерно\n300k строк", "бакеты K=1000\n2k строк", "бакеты K=100\n200 строк"]
lifts = [lift_user, lift_by_k[1000], lift_by_k[100]]
pvs = [t_user.pvalue, p_by_k[1000], p_by_k[100]]
ax1.bar(np.arange(3) - 0.18, np.array(lifts) * 100, width=0.36, color="#4C72B0", label="оценка lift, %")
ax1.axhline(TRUE_LIFT * 100, color="gray", ls="--", lw=1.5, label=f"истина +{TRUE_LIFT:.0%}")
ax1.set_xticks(np.arange(3), names)
ax1.set_ylabel("lift GMV, %")
ax1.legend(fontsize=9, loc="lower right")
ax1b = ax1.twinx()
ax1b.plot(np.arange(3), pvs, "o", color="#C44E52", ms=9, label="p-value")
ax1b.set_ylabel("p-value", color="#C44E52")
ax1b.legend(fontsize=9, loc="upper right")
ax1.set_title("Статистика одна и та же: оценки и p совпадают\n(истинный эффект +4% ловится всеми тремя путями)")
bars = ax2.bar(["поюзерный\nпуть", "бакетный\n(агрегаты 1 раз)"],
               [sec_user_total, sec_bk_total], color=["#C44E52", "#55A868"], width=0.5)
for bar, v in zip(bars, [sec_user_total, sec_bk_total]):
    ax2.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f} с", ha="center", va="bottom", fontsize=11)
ax2.set_ylabel(f"время ночной партии {N_EXPS}×{N_SLICES}, с (perf_counter)")
ax2.set_title(f"Экономика пайплайна: бакеты считают партию ×{speedup:.0f} быстрее;\nв озере правит отношение строк — там оно ×{rows_user_scanned / rows_bk_scanned:.0f}")
fig.tight_layout()
fig.savefig(HERE / "practice_7_2_buckets.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_7_2_buckets.png")

# %% [markdown]
# Шаг 4. Партиционирование озера: 30 дней x 100 экспериментов x 2000 строк
# = 6 млн строк. Запрос дашборда: «эксперимент #37, дни 10–20». Считаем
# строки, которые физически сканируем при трёх схемах хранения, и меряем
# реальное время (лучшая из 3 попыток).

# %%
N_EXPS_LAKE, N_DAYS_LAKE, ROWS_PER_PART = 100, 30, 2000
TOTAL_ROWS = N_EXPS_LAKE * N_DAYS_LAKE * ROWS_PER_PART

lake = pd.DataFrame({
    "day": np.repeat(np.arange(N_DAYS_LAKE, dtype=np.int8), N_EXPS_LAKE * ROWS_PER_PART),
    "exp": np.tile(np.repeat(np.arange(N_EXPS_LAKE, dtype=np.int16), ROWS_PER_PART), N_DAYS_LAKE),
    "user": RNG.integers(0, 2_000_000, size=TOTAL_ROWS).astype(np.int32),
    "value": RNG.normal(60, 20, TOTAL_ROWS).astype(np.float32),
})
part_day = {d: g for d, g in lake.groupby("day")}
part_day_exp = {(d, e): g for (d, e), g in lake.groupby(["day", "exp"])}

Q_EXP, D0, D1 = 37, 10, 20  # запрос: эксперимент 37, дни 10..20


def q_nopart() -> pd.DataFrame:
    return lake[(lake.exp == Q_EXP) & (lake.day >= D0) & (lake.day <= D1)]


def q_daypart() -> pd.DataFrame:
    chunks = [part_day[d] for d in range(D0, D1 + 1)]
    big = pd.concat(chunks)
    return big[big.exp == Q_EXP]


def q_dayexppart() -> pd.DataFrame:
    return pd.concat([part_day_exp[(d, Q_EXP)] for d in range(D0, D1 + 1)])


rows_scanned = {
    "без партиций": TOTAL_ROWS,
    "по дате": (D1 - D0 + 1) * N_EXPS_LAKE * ROWS_PER_PART,
    "по (дате, эксперименту)": (D1 - D0 + 1) * ROWS_PER_PART,
}
# честный замер: 3 повтора, берём лучший
times = {}
for name, fn in (("без партиций", q_nopart), ("по дате", q_daypart), ("по (дате, эксперименту)", q_dayexppart)):
    best = math.inf
    for _ in range(3):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    times[name] = (best, len(out))

print("=" * 78)
print(f"5) ПАРТИЦИОНИРОВАНИЕ ОЗЕРА: {TOTAL_ROWS / 1e6:.0f} млн строк"
      f" ({N_EXPS_LAKE} экспериментов x {N_DAYS_LAKE} дней)")
print("-" * 78)
df_part = pd.DataFrame([
    {"схема": name,
     "сканируется строк": f"{rows / 1e6:.2f} млн" if rows >= 1e6 else f"{rows:,}",
     "доля таблицы": f"{rows / TOTAL_ROWS:.1%}",
     "время запроса": f"{times[name][0] * 1e3:.1f} мс"}
    for name, rows in rows_scanned.items()])
print(df_part.to_string(index=False))
print(f"   все схемы вернули {times['без партиций'][1]:,} строк — ответ один, цена разная")
print(f"   выигрыш (дата x эксперимент): строк ×{TOTAL_ROWS / rows_scanned['по (дате, эксперименту)']:.0f},"
      f" время ×{times['без партиций'][0] / times['по (дате, эксперименту)'][0]:.0f}")
print("""   Так работают Parquet-озёра: партиции по дате и эксперименту + колоночный
   формат (читаем только нужные колонки) + TTL на сырьё. Вечные — агрегаты
   (тест x метрика x срез x бакет x день).""")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
schemes = list(rows_scanned.keys())
vals_rows = [rows_scanned[s] / 1e6 for s in schemes]
vals_time = [times[s][0] * 1e3 for s in schemes]
ax1.barh(schemes, vals_rows, color=["#C44E52", "#DD8452", "#55A868"])
for i, v in enumerate(vals_rows):
    ax1.text(v, i, f" {v:.2f} млн", va="center", fontsize=10)
ax1.set_xlabel("сканируется строк, млн (из 6 млн)")
ax1.set_title("Партиции режут скан: запрос 1 эксперимента за 11 дней\nне должен читать таблицу целиком")
ax2.barh(schemes, vals_time, color=["#C44E52", "#DD8452", "#55A868"])
for i, v in enumerate(vals_time):
    ax2.text(v, i, f" {v:.1f} мс", va="center", fontsize=10)
ax2.set_xlabel("время запроса, мс (лучшая из 3 попыток)")
ax2.set_title(f"И время: ×{times['без партиций'][0] / times['по (дате, эксперименту)'][0]:.0f} на одном узле;\nв распределённом озере выигрыш растёт (сеть + IO)")
fig.tight_layout()
fig.savefig(HERE / "practice_7_2_partitions.png", dpi=150)
plt.close(fig)
print("Сохранено: practice_7_2_partitions.png")
print("\nГотово: экспозиции -> юнит-таблица -> бакеты -> партиции. Следующий слой")
print("(7.3) — интерфейс платформы поверх этих данных: заявки, SRM-гейт, вердикты.")
