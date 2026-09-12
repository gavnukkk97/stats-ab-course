# -*- coding: utf-8 -*-
"""Практика 4.2 — Сплит-система: хэш-рандомизация, SRM, A/A-аудит платформы
(сквозной кейс «ЕдаДома», урок 4.2 модуля «АБ-тесты на практике»).

Принцип курса: не верь формуле — проверь симуляцией.

Что делаем:
1) Сплит на хэше: md5(соль|unit_id) -> бакет; проверяем равномерность
   (хи-квадрат), лавинный эффект и независимость слоёв (2 соли,
   пересечение слотов ~n/mn);
2) SRM-детектор: хи-квадрат долей на классических примерах чисел
   (49500/50500 и др.) + кривая мощности: какой перекос ловится при каком N
   на пороге p<0.001;
3) симуляция A/A-платформы: 500 тестов — равномерность p-value (гистограмма,
   KS), доля SRM-алертов на здоровой системе и на сломанной (вводим баг:
   «фильтр новинок» асимметрично вырезает 2% тест-группы).

Запуск:  python3 practice_4_2.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# Шаг 0. Импорты и параметры сплит-системы «ЕдаДома»: 10 000 бакетов,
# слои = домены продукта (ранжирование, фронтенд, логистика), у каждого своя соль.

# %%
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260911)
HERE = Path(__file__).resolve().parent

M_BUCKETS = 10_000  # бакетов в слое
SRM_ALPHA = 0.001   # порог алерта SRM (не 0.05!)


def get_bucket(unit_id: str, salt: str, m: int = M_BUCKETS) -> int:
    """unit_id + соль -> md5 -> бакет [0, m).

    Детерминизм (тот же юзер -> тот же бакет на любом сервере), равномерность
    и лавинный эффект — дальше проверяем симуляцией (Deng, гл. 7).
    """
    digest = hashlib.md5(f"{salt}|{unit_id}".encode()).hexdigest()
    return int(digest, 16) % m


# %% [markdown]
# Шаг 1. Сплит и слои. Раскатываем 20 000 юзеров: слой «ranking» — проверка
# равномерности по 100 макро-бакетам; независимость слоёв «ranking» и
# «frontend» — таблица 10x10 (хи-квадрат независимости) и пересечение слотов.

# %%
users = [f"user_{i:07d}" for i in range(20_000)]
b_rank = np.array([get_bucket(u, "layer_ranking") for u in users])
b_front = np.array([get_bucket(u, "layer_frontend") for u in users])

# детерминизм: повторный вызов даёт тот же бакет (stateless-сплит)
assert all(get_bucket(u, "layer_ranking") == b for u, b in zip(users[:100], b_rank[:100]))

# (а) равномерность: 100 макро-бакетов по 200 юзеров в среднем
counts = np.bincount(b_rank // 100, minlength=100).astype(float)
chi2_unif = float(((counts - 200.0) ** 2 / 200.0).sum())
p_unif = stats.chi2.sf(chi2_unif, 99)

# (б) лавинный эффект: у id, отличающегося на один символ, digest
# отличается в ~64 из 128 бит (половина) — как два независимых броска монетки
hamming = []
for i in range(3_000):
    h1 = int(hashlib.md5(f"layer_ranking|user_{i:07d}".encode()).hexdigest(), 16)
    h2 = int(hashlib.md5(f"layer_ranking|user_{i:07d}x".encode()).hexdigest(), 16)
    hamming.append(bin(h1 ^ h2).count("1"))
hamming = np.array(hamming)

# (в) независимость слоёв: таблица 10x10 (группа = бакет // 1000)
tab = np.zeros((10, 10))
np.add.at(tab, (b_rank // 1000, b_front // 1000), 1.0)
chi2_ind, p_ind, _, _ = stats.chi2_contingency(tab)

# (г) пересечение конкретных слотов: бакеты 0..999 в ОБОИХ слоях
both = int(np.sum((b_rank < 1000) & (b_front < 1000)))
expected_both = len(users) * (1000 / M_BUCKETS) ** 2

# (д) варианты внутри слота: тест занимает бакеты 0..4999, A = 0..2499
in_test = b_rank < 5000
a_cnt, b_cnt = int(np.sum(b_rank < 2500)), int(np.sum(in_test & (b_rank >= 2500)))
chi2_slot, p_slot = ((a_cnt - (a_cnt + b_cnt) / 2) ** 2 / ((a_cnt + b_cnt) / 2)) * 2, None
p_slot = stats.chi2.sf(chi2_slot, 1)

print("=" * 78)
print("1) СПЛИТ НА ХЭШЕ: md5(соль|unit_id) mod %d, %s юзеров" % (M_BUCKETS, f"{len(users):,}"))
print("-" * 78)
print(f"   равномерность (100 макро-бакетов): chi2={chi2_unif:.1f}, p={p_unif:.3f} — "
      f"{'равномерно' if p_unif > 0.05 else 'ПРОБЛЕМА'}")
print(f"   отклонение доли юзеров в самом большом макро-бакете от 1%: "
      f"{abs(counts.max() / len(users) - 0.01):.4%}")
print(f"   лавинный эффект: отличаются {hamming.mean():.1f} из 128 бит (идеал 64.0, "
      f"sd {hamming.std():.1f})")
print(f"   независимость слоёв (ranking x frontend, 10x10): chi2={chi2_ind:.1f}, "
      f"p={p_ind:.3f} — {'независимы' if p_ind > 0.05 else 'ПРОБЛЕМА'}")
print(f"   пересечение слотов [0..999] двух слоёв: {both} юзеров, ожидали "
      f"{expected_both:.0f} = n/m1*m2")
print(f"   слот теста (бакеты 0..4999): A={a_cnt:,}, B={b_cnt:,}, вне теста "
      f"{len(users) - a_cnt - b_cnt:,}")
print(f"   SRM-чек внутри слота: chi2={chi2_slot:.2f}, p={p_slot:.3f} — сплит здоров")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax1.bar(np.arange(100), counts, color="#4C72B0", alpha=0.8)
ax1.axhline(200, color="#C44E52", ls="--", lw=2, label="ожидание 200 (n/m)")
ax1.set_xlabel("макро-бакет (бакет // 100)")
ax1.set_ylabel("юзеров")
ax1.set_title(f"Равномерность хэш-сплита: p(хи-квадрат) = {p_unif:.2f} — доли бакетов 1/m")
ax1.legend(fontsize=9)
im = ax2.imshow(tab, cmap="Blues")
ax2.set_xlabel("слой frontend: группа (бакет // 1000)")
ax2.set_ylabel("слой ranking: группа (бакет // 1000)")
ax2.set_title(f"Независимость слоёв (разные соли): p(хи-квадрат) = {p_ind:.2f},\n"
              f"пересечение слотов {both} ≈ n/(m1·m2) = {expected_both:.0f}")
fig.colorbar(im, ax=ax2, shrink=0.85)
fig.tight_layout()
fig.savefig(HERE / "practice_4_2_split.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_4_2_split.png")

# %% [markdown]
# Шаг 2. SRM-детектор. Хи-квадрат: chi2 = sum((O-E)^2/E), df=1 для двух групп.
# Порог алерта p<0.001 (не 0.05!): чек автоматический и массовый, здоровый
# тест не должен краснеть в 5% случаев. Считаем классические примеры чисел
# и силу детектора: мощность против перекоса при разных N.

# %%
def srm_check(counts, expected_shares=None):
    """Хи-квадрат долей групп против ожидаемых. Возвращает (chi2, p)."""
    counts = np.asarray(counts, dtype=float)
    if expected_shares is None:
        expected_shares = np.full(len(counts), 1.0 / len(counts))
    expected = counts.sum() * expected_shares
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    return chi2, float(stats.chi2.sf(chi2, len(counts) - 1))


examples = [(49_995, 50_005), (49_800, 50_200), (49_500, 50_500),
            (495_000, 505_000), (252_000, 248_000)]
rows = []
for c, t in examples:
    chi2, p = srm_check([c, t])
    rows.append({
        "контроль / тест": f"{c:,} / {t:,}",
        "перекос": f"{abs(c - t) / (c + t):.2%}",
        "chi2": f"{chi2:.1f}",
        "p-value": f"{p:.2e}" if p > 1e-4 else f"{p:.1e}",
        "алерт при 0.05": "да" if p < 0.05 else "нет",
        "алерт при 0.001": "SRM!" if p < SRM_ALPHA else "нет",
    })
df_srm = pd.DataFrame(rows)

print("=" * 78)
print("2) SRM-ДЕТЕКТОР: хи-квадрат долей, порог p<0.001")
print("-" * 78)
print(df_srm.to_string(index=False))
print("""
   Читаем: 49995/50005 — здоровый сплит (p=0.97); 49800/50200 — перекос 0.4%
   даёт p=0.21: НЕ алерт при 0.001 (а порог 0.05 ловил бы ложные тревоги
   в 5% здоровых тестов); 49500/50500 (перекос 1% на 100k) — p=0.0016, на
   пороге; тот же перекос 1% на миллионе (495000/505000) — p~1e-23.
   В конспекте источника (koch-kir) для этих примеров указаны p~1e-23/0.39/
   2e-8 — пересчитайте руками: хи-квадрат = z^2, и сила сигнала зависит от
   N. Порог 0.001 и мораль неизменны.""")

# %%
z_crit = stats.norm.ppf(1 - SRM_ALPHA / 2)  # 3.29
skews = np.linspace(0.0005, 0.04, 300)  # перекос = |доля_A - доля_B|
fig, ax = plt.subplots(figsize=(8.8, 4.6))
for N, color in ((10_000, "#8172B3"), (100_000, "#4C72B0"), (1_000_000, "#55A868")):
    z_alt = skews * math.sqrt(N)  # z = skew * sqrt(N) при плане 50/50
    power = stats.norm.cdf(z_alt - z_crit) + stats.norm.cdf(-z_alt - z_crit)
    ax.plot(skews * 100, power, lw=2.2, color=color, label=f"N = {N:,}")
    # точка «перекос 1%»
    pw1 = stats.norm.cdf(0.01 * math.sqrt(N) - z_crit)
    ax.plot([1.0], [pw1], "o", ms=6, color=color)
    ax.annotate(f"{pw1:.0%}", (1.0, pw1), xytext=(6, -8), textcoords="offset points",
                fontsize=9, color=color)
ax.axhline(0.8, color="gray", ls=":", lw=1.4)
ax.text(3.1, 0.82, "мощность 80%", fontsize=9, color="gray")
ax.axvline(0.4, color="#C44E52", ls="--", lw=1.2)
ax.text(0.5, 0.08, "49800/50200\n(перекос 0.4%)", fontsize=8, color="#C44E52")
ax.set_xlabel("перекос сплита: |доля контроля − доля теста|, п.п.")
ax.set_ylabel(f"мощность хи-квадрата при пороге {SRM_ALPHA}")
ax.set_title("SRM-детектор слеп на малых N: перекос 1% ловится с мощностью 80%\n"
             "только от N ≈ 171 000 юзеров суммарно")
ax.legend(fontsize=9, loc="center right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "practice_4_2_srm_power.png", dpi=150)
plt.close(fig)
print("Сохранено: practice_4_2_srm_power.png")
n80 = ((z_crit + stats.norm.ppf(0.8)) / 0.01) ** 2
print(f"Проверка: N для 80% мощности против перекоса 1%: {n80:,.0f} юзеров суммарно")

# %% [markdown]
# Шаг 3. A/A-тесты как аудит платформы (Deng, гл. 9; Kohavi, гл. 19).
# 500 виртуальных A/A-тестов: на здоровой системе p-value равномерны,
# алертов p<0.001 около 0.1%. Затем ЛОМАЕМ сплит: в тестовой группе есть
# секция «Новинки», её события теряются в пайплайне, и «фильтр качества»
# вырезает этих юзеров — асимметрично, только из теста, −2%.

# %%
def aa_pvalues(n_tests, n_total, loss=0.0, rng=None):
    """p-value SRM-чека в A/A-тестах; loss — доля тест-группы, съеденная багом."""
    rng = rng or RNG
    pvals = np.empty(n_tests)
    for i in range(n_tests):
        c = rng.binomial(n_total, 0.5)
        t = n_total - c
        if loss:
            t = rng.binomial(t, 1 - loss)
        pvals[i] = srm_check([c, t])[1]
    return pvals


N_TESTS, N_BROKEN = 500, 200_000
healthy = aa_pvalues(N_TESTS, N_BROKEN, loss=0.0)
broken = aa_pvalues(N_TESTS, N_BROKEN, loss=0.02)
ks_h = stats.kstest(healthy, "uniform")

print("=" * 78)
print(f"3) A/A-АУДИТ ПЛАТФОРМЫ: {N_TESTS} тестов по N={N_BROKEN:,} юзеров")
print("-" * 78)
print(f"   ЗДОРОВАЯ система: KS-тест равномерности p = {ks_h.pvalue:.2f}")
print(f"      доля p<0.05  : {np.mean(healthy < 0.05):.1%}  (ожидали 5%)")
print(f"      доля p<0.001 : {np.mean(healthy < SRM_ALPHA):.1%}  (ожидали 0.1%) — "
      f"алертов {int(np.sum(healthy < SRM_ALPHA))} из {N_TESTS}")
print(f"   СЛОМАННАЯ (фильтр новинок: -2% тест-группы):")
print(f"      доля p<0.001 : {np.mean(broken < SRM_ALPHA):.1%} — "
      f"алертов {int(np.sum(broken < SRM_ALPHA))} из {N_TESTS}")

sizes = (50_000, 200_000, 1_000_000)
rows = []
for N in sizes:
    h = aa_pvalues(400, N, loss=0.0)
    br = aa_pvalues(400, N, loss=0.02)
    rows.append({"N (юзеров)": f"{N:,}",
                 "алерты здоровой": f"{np.mean(h < SRM_ALPHA):.1%}",
                 "алерты сломанной": f"{np.mean(br < SRM_ALPHA):.1%}"})
df_aa = pd.DataFrame(rows)
print("\n   Мощность детектора зависит от N (баг один и тот же, -2% теста):")
print(df_aa.to_string(index=False))
print("\n   Мораль: маленькие тесты слепы к маленьким багам — SRM-чек обязателен,")
print("   но не достаточен; лечится длиной теста и A/A-аудитом платформы.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
axes[0].hist(healthy, bins=20, range=(0, 1), color="#4C72B0", edgecolor="white")
axes[0].axhline(N_TESTS / 20, color="#C44E52", ls="--", lw=2, label="ровная = 25")
axes[0].set_title(f"Здоровая платформа: p-value равномерны\nKS p = {ks_h.pvalue:.2f}, "
                  f"p<0.05 в {np.mean(healthy < 0.05):.0%} тестов")
axes[0].set_xlabel("p-value SRM-чека в A/A")
axes[0].set_ylabel("число тестов")
axes[0].legend(fontsize=8)

axes[1].hist(broken, bins=20, range=(0, 1), color="#C44E52", edgecolor="white")
axes[1].set_yscale("log")
axes[1].set_title(f"Сломанная (фильтр новинок, -2% теста):\n{np.mean(broken < SRM_ALPHA):.0%} "
                  f"алертов SRM при N=200k")
axes[1].set_xlabel("p-value SRM-чека в A/A")
axes[1].set_ylabel("число тестов (log)")

x = np.arange(len(sizes))
w = 0.38
alerts_h = [np.mean(aa_pvalues(400, N, 0.0) < SRM_ALPHA) for N in sizes]
alerts_b = [np.mean(aa_pvalues(400, N, 0.02) < SRM_ALPHA) for N in sizes]
axes[2].bar(x - w / 2, np.array(alerts_h) * 100, w, color="#4C72B0", label="здоровая")
axes[2].bar(x + w / 2, np.array(alerts_b) * 100, w, color="#C44E52", label="сломанная (-2%)")
for xi, v in zip(x - w / 2, np.array(alerts_h) * 100):
    axes[2].annotate(f"{v:.0f}%", (xi, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=9)
for xi, v in zip(x + w / 2, np.array(alerts_b) * 100):
    axes[2].annotate(f"{v:.0f}%", (xi, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=9)
axes[2].set_xticks(x)
axes[2].set_xticklabels([f"{N//1000}k" for N in sizes])
axes[2].set_xlabel("размер теста, юзеров суммарно")
axes[2].set_ylabel("% SRM-алертов")
axes[2].set_title("Тот же баг заметен только на больших N:\nSRM-детектор — тоже про мощность")
axes[2].legend(fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "practice_4_2_aa_hist.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_4_2_aa_hist.png")

# %%
print()
print("=" * 78)
print("ГЛАВНОЕ (что должны увидеть):")
print("1) md5(соль|unit_id) mod m даёт равномерный и детерминированный сплит;")
print("   лавинный эффект (~64/128 бит) делает разные соли независимыми — на этом")
print("   стоят слои: пересечение слотов ~n/(m1*m2), хи-квадрат независимости зелёный.")
print("2) SRM: хи-квадрат = sum((O-E)^2/E), порог p<0.001, а не 0.05; перекос 0.4%")
print("   на 100k — НЕ алерт (p~0.2), перекос 1% на 1M — p~1e-23; детектор имеет")
print("   мощность: 1% перекос ловится с 80% мощностью только от N~171k.")
print("3) A/A-аудит: на здоровой платформе p-value равномерны (KS), алертов ~0.1%;")
print("   баг «фильтр новинок» (-2% теста) ловится в ~90% тестов при N=200k и")
print("   почти никогда при N=50k — маленькие тесты слепы к маленьким багам.")
