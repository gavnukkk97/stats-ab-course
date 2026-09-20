# -*- coding: utf-8 -*-
"""Практика 1.4 — Бутстрап (сквозной кейс «ЕдаДома»).

Не верь — проверь симуляцией.

Что делаем:
1) бутстрап-интервалы (B = 5000, бут-выборка объёма n) для медианы чека
   и 95-го перцентиля — статистик, для которых нет простых формул SE;
2) сравниваем с классическим t-интервалом для среднего
   (сам t-тест — урок 2.2, здесь только интервалы);
3) показываем «ложное сужение»: бут-выборки объёма n/5 и 5n искажают
   ширину интервала в ~sqrt(5) раз (правило Atlamos: бут-выборка строго объёма n);
4) QQ-plot бутстрап-распределения медианы — оно несимметрично,
   поэтому процентильный интервал честнее x̄ ± 1.96·SE.

Запуск из корня репозитория: python3 course/modules/M1/practice/practice_1_4.py
Графики сохраняются в папку images/ этого модуля.
"""

# %% [markdown]
# Шаг 0. Импорты и данные: 500 чеков «ЕдаДома» (логнормал, как в уроке 1.3).
# Истинные параметры совокупности нам известны — но бутстрап их не видит:
# он работает ТОЛЬКО с выборкой (выборка — суррогат совокупности).

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))
from foundations import normal_power, type_m, mc_interval, mc_report

RNG = np.random.default_rng(73)  # seed -> результат воспроизводим
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

B = 5_000            # число бутстрап-реплик (симулятор АБ: 1000–10000)
N_ORDERS = 500       # объём выборки (и, по правилу, каждой бут-выборки)
CONF_Q = (0.025, 0.975)

MEDIAN_TRUE = 800.0  # истинная медиана чека, руб.
SIGMA_LOG = 1.1
checks = MEDIAN_TRUE * RNG.lognormal(0.0, SIGMA_LOG, size=N_ORDERS)
TRUE_MEAN = MEDIAN_TRUE * np.exp(SIGMA_LOG**2 / 2)

mean_hat = checks.mean()
median_hat = np.median(checks)
p95_hat = np.quantile(checks, 0.95)

print("=" * 80)
print("0) Выборка «ЕдаДома»")
print(f"   n = {N_ORDERS} заказов")
print(f"   выборка:  mean = {mean_hat:,.0f} руб., median = {median_hat:,.0f} руб., "
      f"p95 = {p95_hat:,.0f} руб.")
print(f"   истина:   mean = {TRUE_MEAN:,.0f} руб., median = {MEDIAN_TRUE:,.0f} руб. "
      f"(бутстрапу недоступна)")


def bootstrap_stat(data, stat_fn, B, m, rng, chunk=500):
    """B бутстрап-реплик статистики по бут-выборкам объёма m.

    Правильно: m = len(data). m != n — сознательная ошибка шага 3.
    """
    n = len(data)
    out = np.empty(B)
    for start in range(0, B, chunk):
        k = min(chunk, B - start)
        idx = rng.integers(0, n, size=(k, m))  # индексы С ВОЗВРАЩЕНИЕМ
        out[start:start + k] = stat_fn(data[idx])
    return out


median_fn = lambda a: np.median(a, axis=-1)
p95_fn = lambda a: np.quantile(a, 0.95, axis=-1)

# %% [markdown]
# Шаг 1. Корректный бутстрап: бут-выборка объёма n, B = 5000 реплик,
# процентильный интервал = 2.5-й и 97.5-й перцентили бут-распределения.

# %%
boot_median = bootstrap_stat(checks, median_fn, B, N_ORDERS, RNG)
boot_p95 = bootstrap_stat(checks, p95_fn, B, N_ORDERS, RNG)
ci_med = np.quantile(boot_median, CONF_Q)
ci_p95 = np.quantile(boot_p95, CONF_Q)

# t-интервал для среднего — для сравнения (t-тест ждёт нас в М2)
se_mean = checks.std(ddof=1) / np.sqrt(N_ORDERS)
t_crit = stats.t.ppf(0.975, df=N_ORDERS - 1)
ci_mean = (mean_hat - t_crit * se_mean, mean_hat + t_crit * se_mean)

print("=" * 80)
print("1) Доверительные интервалы (95%)")
print(f"   {'статистика':<28}{'оценка':>12}{'CI (бутстрап/t)':>26}{'ширина':>10}")
print(f"   {'медиана чека':<28}{median_hat:>12,.0f}"
      f"{f'[{ci_med[0]:,.0f}; {ci_med[1]:,.0f}]':>26}"
      f"{ci_med[1] - ci_med[0]:>10,.0f}")
print(f"   {'95-й перцентиль чека':<28}{p95_hat:>12,.0f}"
      f"{f'[{ci_p95[0]:,.0f}; {ci_p95[1]:,.0f}]':>26}"
      f"{ci_p95[1] - ci_p95[0]:>10,.0f}")
print(f"   {'среднее чека (t-CI)':<28}{mean_hat:>12,.0f}"
      f"{f'[{ci_mean[0]:,.0f}; {ci_mean[1]:,.0f}]':>26}"
      f"{ci_mean[1] - ci_mean[0]:>10,.0f}")
print(f"   Покрыта ли истинная медиана: {ci_med[0] <= MEDIAN_TRUE <= ci_med[1]}")

# %% [markdown]
# Шаг 2. Ловушка объёма бут-выборки (Atlamos): бут-выборки n/5 и 5n.
# Бут-выборка объёма m имитирует «эксперимент» с m наблюдениями: её SE ~ 1/√m.
# m = 5n сужает интервал в ~√5 раз — ЛОЖНАЯ точность (Atlamos разбирает 25n:
# сужение в 5 раз); m = n/5 наоборот расширяет в ~√5 раз — имитирует
# эксперимент впятеро меньше того, что мы реально провели.

# %%
boot_med_small = bootstrap_stat(checks, median_fn, B, N_ORDERS // 5, RNG)
boot_med_large = bootstrap_stat(checks, median_fn, B, 5 * N_ORDERS, RNG)
ci_small = np.quantile(boot_med_small, CONF_Q)
ci_large = np.quantile(boot_med_large, CONF_Q)
w_ok = ci_med[1] - ci_med[0]
w_small = ci_small[1] - ci_small[0]
w_large = ci_large[1] - ci_large[0]

print("=" * 80)
print("2) Ложное сужение/расширение при неверном объёме бут-выборки (медиана)")
print(f"   {'объём бут-выборки':<24}{'CI медианы':>24}{'ширина':>10}{'к верной':>10}")
for label, ci, w in (
    ("m = n/5 = 100", ci_small, w_small),
    ("m = n = 500 (верно)", ci_med, w_ok),
    ("m = 5n = 2500", ci_large, w_large),
):
    print(f"   {label:<24}{f'[{ci[0]:,.0f}; {ci[1]:,.0f}]':>24}{w:>10,.0f}{w / w_ok:>10.2f}")
print(f"   Теория: ширина ~ 1/√m -> n/5 ширит в √5 = {np.sqrt(5):.2f} раза,")
print("   5n сужает в √5 раз (у Atlamos 25n -> сужение в 5 раз) — ложная точность.")
print("   Вывод: бут-выборка СТРОГО объёма n — иначе дисперсия искажена.")

# %% [markdown]
# Шаг 3. QQ-plot бут-распределения медианы против нормального.
# При n = 500 оно уже почти нормально (ЦПТ работает и для квантилей),
# но лёгкая асимметрия в хвостах остаётся — процентильный интервал
# учитывает её автоматически, «оценка ± 1.96·SE» — нет.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

ax = axes[0]
ax.hist(checks, bins=60, color="#8172B2", alpha=0.8)
ax.axvline(mean_hat, color="#C44E52", lw=2, label=f"mean = {mean_hat:,.0f}")
ax.axvline(median_hat, color="#4C72B0", lw=2, label=f"median = {median_hat:,.0f}")
ax.axvline(p95_hat, color="#55A868", lw=2, ls="--", label=f"p95 = {p95_hat:,.0f}")
ax.set_xlim(0, np.quantile(checks, 0.99))
ax.set_title("Выборка чеков (n = 500)")
ax.set_xlabel("чек, руб.")
ax.set_ylabel("частота")
ax.legend(fontsize=8)

for ax, boots, hat, ci, title in (
    (axes[1], boot_median, median_hat, ci_med, "Бутстрап: медиана"),
    (axes[2], boot_p95, p95_hat, ci_p95, "Бутстрап: p95"),
):
    ax.hist(boots, bins=60, density=True, color="#4C72B0", alpha=0.8)
    ax.axvline(hat, color="k", ls="--", lw=1.5, label="точечная оценка")
    ax.axvline(ci[0], color="#C44E52", lw=2)
    ax.axvline(ci[1], color="#C44E52", lw=2,
               label=f"95% CI = [{ci[0]:,.0f}; {ci[1]:,.0f}]")
    ax.set_title(f"{title}, B = {B}")
    ax.set_xlabel("значение статистики на бут-выборке")
    ax.legend(fontsize=8)

fig.suptitle("Бутстрап «ЕдаДома»: интервалы для статистик без формул", fontsize=13)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "bootstrap_checks.png", dpi=150)
print("Сохранено: bootstrap_checks.png")

# %%
fig, ax = plt.subplots(figsize=(6.2, 6.2))
(osm, osr), (slope, intercept, r) = stats.probplot(boot_median, dist="norm")
ax.plot(osm, osr, ".", ms=3, color="#4C72B0")
ax.plot(osm, slope * osm + intercept, "-", color="#C44E52", lw=2)
ax.set_title("QQ-plot: бутстрап-распределение медианы")
ax.set_xlabel("теоретические квантили N(0, 1)")
ax.set_ylabel("квантили бутстрап-медиан")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "bootstrap_qq_median.png", dpi=150)

skew_boot_med = stats.skew(boot_median)
skew_boot_p95 = stats.skew(boot_p95)
print("=" * 80)
print("3) Форма бут-распределений")
print(f"   медиана: асимметрия = {skew_boot_med:.2f}, R² против нормального = {r**2:.3f}")
print(f"   p95:     асимметрия = {skew_boot_p95:.2f} — у хвостовых статистик изгиб сильнее")
print("   На QQ-plot медианы точки чуть отрываются от прямой в хвостах:")
print("   небольшая асимметрия, которую процентильный CI учитывает,")
print("   а «оценка ± 1.96·SE» — нет (при меньших n разница заметнее).")

# %%
print()
print("=" * 80)
print("ГЛАВНОЕ (что должны увидеть):")
print(f"1) Медиана: {median_hat:,.0f} руб., 95% CI [{ci_med[0]:,.0f}; {ci_med[1]:,.0f}] "
      f"— покрытие истинной медианы: {ci_med[0] <= MEDIAN_TRUE <= ci_med[1]}.")
print(f"2) p95: {p95_hat:,.0f} руб., CI [{ci_p95[0]:,.0f}; {ci_p95[1]:,.0f}] — "
      f"шире, чем у медианы: хвост оценивается грубее.")
print(f"3) t-CI среднего [{ci_mean[0]:,.0f}; {ci_mean[1]:,.0f}] шире бут-CI медианы:")
print("   это разные параметры: меньшая ширина не доказывает лучшую бизнес-метрику.")
print(f"4) m = 5n -> ширина {w_large / w_ok:.2f} от верной (≈1/√5): сужение в "
      f"{w_ok / w_large:.1f} раза — ложная точность")
print(f"   (Atlamos: бут-выборки 25n сужают распределение в 5 раз);")
print(f"   m = n/5 -> ширина {w_small / w_ok:.2f}× (≈√5) — имитация меньшего эксперимента.")
print("5) Бут-выборка всегда объёма n; B = 1000–10000; юниты — целиком (М3.3).")

# Coverage requires NEW original samples, not repeated resampling of one sample.
coverage_rng = np.random.default_rng(1404)
worlds_coverage, bootstrap_replicates = 300, 999
coverage = {method: 0 for method in ("percentile", "basic", "BCa")}
for _ in range(worlds_coverage):
    sample = coverage_rng.lognormal(np.log(800), 1.1, size=80)
    for method in coverage:
        result = stats.bootstrap((sample,), np.median, method=method, n_resamples=bootstrap_replicates, rng=coverage_rng)
        lo, hi = result.confidence_interval
        coverage[method] += lo <= 800 <= hi
for method, count in coverage.items():
    print("Median coverage", method, mc_report(count, worlds_coverage))
print("300 миров дают лишь грубое сравнение покрытия; близкие доли не доказывают превосходство метода.")
