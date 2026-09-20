from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "course" / "modules" / "M3" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Визуализации для М3 «Статистика для АБ-тестов»
# Единый стиль: заголовок = вывод. Запуск: python3 scripts/figures/make_visuals_m3.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

rng = np.random.default_rng(303)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_ACC, C_TRUE, C_BAD = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

# ============================================================
# 32. Ошибка ×2: одновыборочная vs двухвыборочная формула
# ============================================================
mde = np.linspace(0.01, 0.15, 200)
sigma, za, zb = 1.0, 1.96, 0.84
n_one = (za + zb) ** 2 * sigma ** 2 / mde ** 2   # (без множителя 2 на группу)
n_two = 2 * (za + zb) ** 2 * sigma ** 2 / mde ** 2
fig, ax = plt.subplots(figsize=(8, 4.3))
ax.plot(mde * 100, n_two, lw=2.6, color=C_TRUE, label="правильно: двухвыборочная, на руку 2·(z₊+z₋)²σ²/MDE²")
ax.plot(mde * 100, n_one, lw=2.6, color=C_BAD, ls="--", label="типичная ошибка: без ×2 — выборки вдвое меньше")
ax.axvline(3, color="gray", ls=":", lw=1.4)
ax.annotate("MDE=3%: 17 422 нужно\nзабыли ×2 → 8 711", xy=(3, 17500), xytext=(5.5, 30000),
            arrowprops=dict(arrowstyle="->", color=C_BAD), color=C_BAD, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffe3e3"))
ax.set_yscale("log")
ax.set_xlabel("MDE, % от σ"); ax.set_ylabel("наблюдений на руку (log)")
ax.set_title("Самая дорогая опечатка в планировании: тест двухвыборочный — множитель 2 обязателен")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v32_mde_factor2.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 33. Карта power(MDE, n)
# ============================================================
mde = np.linspace(0.01, 0.12, 120)
ns = [500, 2000, 8000, 32000]
fig, ax = plt.subplots(figsize=(8, 4.4))
for n in ns:
    signal = mde * np.sqrt(n / 2)
    pw = stats.norm.cdf(signal-1.96) + stats.norm.cdf(-signal-1.96)
    ax.plot(mde * 100, pw, lw=2.4, label=f"n на руку = {n:,}".replace(",", " "))
ax.axhline(0.8, color="gray", ls=":", lw=1.4); ax.text(11.3, 0.82, "power 80%", fontsize=9)
ax.set_xlabel("MDE, % (в единицах σ метрики)"); ax.set_ylabel("мощность")
ax.set_title("Треугольник дизайнера: n × MDE × мощность — задай любые два, третье получится")
ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v33_power_map.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 34. Сплит 50/50 vs 10/90
# ============================================================
n_total, mde_s = 20000, 0.05
rs = np.linspace(0.05, 0.95, 180)
eff = mde_s / np.sqrt(rs * (1 - rs))  # SE разности ~ σ*sqrt(1/(r(1-r)n))
pw = stats.norm.cdf(mde_s * np.sqrt(n_total * rs * (1 - rs)) - 1.96)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(rs * 100, pw, lw=2.6, color=C_MAIN)
ax.scatter([50, 10], [stats.norm.cdf(mde_s * np.sqrt(n_total * 0.25) - 1.96),
                      stats.norm.cdf(mde_s * np.sqrt(n_total * 0.09) - 1.96)],
           s=90, color=[C_TRUE, C_BAD], zorder=5)
ax.annotate(f"50/50: power {stats.norm.cdf(mde_s*np.sqrt(5000)-1.96):.1%}", xy=(50, stats.norm.cdf(mde_s*np.sqrt(5000)-1.96)),
            xytext=(28, 0.9), color=C_TRUE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_TRUE))
ax.annotate(f"10/90: power {stats.norm.cdf(mde_s*np.sqrt(1800)-1.96):.1%}\n(при том же трафике!)", xy=(10, stats.norm.cdf(mde_s*np.sqrt(1800)-1.96)),
            xytext=(18, 0.35), color=C_BAD, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_BAD))
ax.set_xlabel("доля тестовой группы, %"); ax.set_ylabel("мощность")
ax.set_title("При равных дисперсиях 50/50 мощнее: r·(1−r) максимален при 50/50")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v34_split_power.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 35. Юнит рандомизации: юзер vs заказ
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
users_a = rng.normal(0, 1, 14)
orders_a = {u: rng.normal(u * 0.55, 0.6, size=k) for u, k in zip(users_a, rng.integers(1, 6, 14))}
for u, os_ in orders_a.items():
    axes[0].scatter(np.full(len(os_), u) + rng.uniform(-0.06, 0.06, len(os_)), os_,
                    s=30, color=C_MAIN, alpha=0.85)
axes[0].set_xlabel("юзер (рандомизировали его!)"); axes[0].set_ylabel("заказ")
axes[0].set_title("Юнит рандомизации = юзер:\nзаказы одного юзера — гроздью вокруг «своего» уровня", fontsize=10)
all_orders = np.concatenate(list(orders_a.values()))
axes[1].scatter(rng.uniform(-0.2, 0.2, len(all_orders)), all_orders, s=30, color=C_BAD, alpha=0.6)
axes[1].set_xticks([]); axes[1].set_xlabel("«просто заказы»")
axes[1].set_title("Тот же тест, анализ по заказам:\nзависимость потеряна → SE занижен → ложные значимости", fontsize=10)
fig.suptitle("Кластеры юзера: рандомизация даёт независимость ЮНИТАМ, а не заказам",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v35_randomization_unit.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 36. ICC и design effect
# ============================================================
ms = np.arange(1, 41)
fig, ax = plt.subplots(figsize=(8, 4.2))
for icc, c in [(0.01, C_TRUE), (0.05, C_MAIN), (0.2, C_BAD)]:
    de = 1 + (ms - 1) * icc
    ax.plot(ms, de, lw=2.4, color=c, label=f"ICC = {icc}")
ax.set_xlabel("средний размер кластера m (заказов на юзера)")
ax.set_ylabel("design effect = 1+(m−1)·ICC")
ax.set_title("Во сколько раз раздувается дисперсия: считай эффективный n = n/DE")
ax.legend()
ax.text(0.98, 0.35, "кластер 20 заказов, ICC=0.05:\nDE=1.95 — половина выборки «сгорела»",
        transform=ax.transAxes, ha="right", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v36_icc_design_effect.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 37. Ratio-метрика: числитель/знаменатель
# ============================================================
n = 60
orders = rng.poisson(2.2, n)
rev = np.array([rng.lognormal(np.log(900), 0.8, max(k, 1)).sum() if k else 0 for k in orders])
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.scatter(orders + rng.uniform(-0.12, 0.12, n), rev, s=55, color=C_MAIN, alpha=0.8)
ax.set_xlabel("заказов (знаменатель)"); ax.set_ylabel("выручка (числитель)")
ax.set_title("Средний чек = Σвыручка/Σзаказы: числитель и знаменатель скоррелированы\n— поэтому t-тест «по заказам» против правил")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v37_ratio_num_denom.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 38. Четыре способа тестировать ratio: уровень в A/A
# ============================================================
def ratio_aa(method, reps=1500, n=300):
    rejected = 0
    for _ in range(reps):
        groups = []
        orders = []
        for _arm in range(2):
            count = rng.poisson(2.2, n)
            user = rng.normal(0, .7, n)
            events = [rng.lognormal(np.log(900)+u, .5, int(k)) for k,u in zip(count,user)]
            revenue = np.array([values.sum() for values in events])
            groups.extend([count, revenue])
            orders.append(np.concatenate(events))
        p = stats.ttest_ind(*orders, equal_var=False).pvalue if method == "naive" else method(*groups)
        rejected += p < .05
    return rejected / reps

def boot_users(o1, r1, o2, r2, B=400):
    stat0 = r1.sum() / o1.sum() - r2.sum() / o2.sum()
    boots = []
    for _ in range(B):
        i1 = rng.integers(0, len(r1), len(r1)); i2 = rng.integers(0, len(r2), len(r2))
        boots.append(r1[i1].sum() / np.maximum(o1[i1].sum(), 1) - r2[i2].sum() / np.maximum(o2[i2].sum(), 1))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return 1.0 if lo <= 0 <= hi else 0.0

def delta(o1, r1, o2, r2):
    mu_n, mu_d = r1.mean(), o1.mean()
    var_n, var_d = r1.var(ddof=1), o1.var(ddof=1)
    cov = np.cov(r1, o1, ddof=1)[0, 1]
    v1 = var_n / mu_d ** 2 - 2 * mu_n * cov / mu_d ** 3 + mu_n ** 2 * var_d / mu_d ** 4
    se1 = np.sqrt(v1 / len(r1))
    mu_n2, mu_d2 = r2.mean(), o2.mean()
    var_n2, var_d2 = r2.var(ddof=1), o2.var(ddof=1)
    cov2 = np.cov(r2, o2, ddof=1)[0, 1]
    v2 = var_n2 / mu_d2 ** 2 - 2 * mu_n2 * cov2 / mu_d2 ** 3 + mu_n2 ** 2 * var_d2 / mu_d2 ** 4
    se2 = np.sqrt(v2 / len(r2))
    z = (r1.sum() / o1.sum() - r2.sum() / o2.sum()) / np.sqrt(se1 ** 2 + se2 ** 2)
    return 2 * (1 - stats.norm.cdf(abs(z)))

def lin(o1, r1, o2, r2):
    rc = r2.sum() / o2.sum()
    s1 = r1 - rc * o1
    s2 = r2 - rc * o2
    return stats.ttest_ind(s1, s2, equal_var=False).pvalue

lv_naive = ratio_aa("naive", reps=1200)
lv_boot = ratio_aa(boot_users, reps=1200)
lv_delta = ratio_aa(delta, reps=1500)
lv_lin = ratio_aa(lin, reps=1500)
fig, ax = plt.subplots(figsize=(8.4, 4.2))
vals = [lv_naive, lv_boot, lv_delta, lv_lin]
cols = [C_BAD if v > 0.075 else C_TRUE for v in vals]
ax.bar(range(4), np.array(vals) * 100, color=cols, alpha=0.9, width=0.6)
ax.axhline(5, color="black", ls="--", lw=1.6)
ax.set_xticks(range(4))
ax.set_xticklabels(["t по заказам\n(наивный)", "бутстрап\nюзерами", "дельта-метод", "линеаризация"])
for i, v in enumerate(vals):
    ax.text(i, v * 100 + 0.25, f"{v*100:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("доля p<0.05 в A/A, %")
ax.set_title("Ratio-метрика в A/A: наивный t-тест ломает уровень, три правильных способа держат 5%")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v38_ratio_aa_level.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 39–41. Винзоризация: три сюжета
# ============================================================
x = rng.lognormal(np.log(900), 1.1, 6000)
lo, hi = np.percentile(x, [1, 99])
xw = np.clip(x, lo, hi)
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
axes[0].hist(x, bins=150, color=C_MAIN, alpha=0.85)
axes[0].axvline(hi, color=C_BAD, lw=2.4)
axes[0].set_title(f"Сырой чек: SD = {x.std():,.0f} ₽".replace(",", " "), fontsize=10)
axes[1].hist(xw, bins=80, color=C_ACC, alpha=0.9)
axes[1].axvline(hi, color=C_BAD, lw=2.4, ls="--")
axes[1].set_title(f"Винзоризация 1/99: SD = {xw.std():,.0f} ₽ (−{100*(1-xw.std()/x.std()):.0f}%)".replace(",", " "), fontsize=10)
for ax_ in axes:
    ax_.set_xlabel("чек, ₽")
axes[0].text(0.98, 0.55, f"порог P99 = {hi:,.0f} ₽".replace(",", " "), transform=axes[0].transAxes,
             ha="right", fontsize=10, color=C_BAD, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
fig.suptitle("Винзоризация сжимает дисперсию — тесты становятся чувствительнее",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v39_winsorization_sd.png", bbox_inches="tight"); plt.close(fig)

# 40: эффект в хвосте исчезает
n = 2000
base = rng.lognormal(np.log(900), 1.1, n)
treat = base * np.where(base > np.percentile(base, 90), 1.5, 1.0)
fig, ax = plt.subplots(figsize=(8.4, 4.3))
lo2, hi2 = np.percentile(np.concatenate([base, treat]), [1, 99])
d_raw = treat.mean() - base.mean()
d_win = np.clip(treat, lo2, hi2).mean() - np.clip(base, lo2, hi2).mean()
p_raw = stats.ttest_ind(treat, base, equal_var=False).pvalue
p_win = stats.ttest_ind(np.clip(treat, lo2, hi2), np.clip(base, lo2, hi2), equal_var=False).pvalue
bars = ax.bar([0, 1], [d_raw, d_win], color=[C_BAD, C_MAIN], width=0.5, alpha=0.9)
ax.text(0, d_raw + 8, f"raw: эффект {d_raw:.0f} ₽\np = {p_raw:.3f}", ha="center", fontweight="bold")
ax.text(1, d_win + 8, f"винзоризация: {d_win:.0f} ₽\np = {p_win:.3f} — эффект СКРЫТ", ha="center",
        fontweight="bold", color=C_BAD)
ax.set_xticks([0, 1]); ax.set_xticklabels(["сырые данные", "винзоризация 1/99"])
ax.set_ylabel("оценка эффекта, ₽")
ax.set_title("Эффект живёт в хвосте (киты ×1.5): винзоризация спрячет реальный эффект\n«не отвергли H0» ≠ «эффекта нет»")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v40_winsor_hides_effect.png", bbox_inches="tight"); plt.close(fig)

# 41: раздельные трешолды → смещение
fig, ax = plt.subplots(figsize=(8.4, 4.2))
bias_common, bias_sep = [], []
for _ in range(1200):
    a = rng.lognormal(np.log(900), 1.1, 800)
    b = rng.lognormal(np.log(900), 1.1, 800) * 1.0
    t_hi = np.percentile(np.concatenate([a, b]), 99)
    a_hi, b_hi = np.percentile(a, 99), np.percentile(b, 99)
    bias_common.append(np.clip(a, None, t_hi).mean() - np.clip(b, None, t_hi).mean())
    bias_sep.append(np.clip(a, None, a_hi).mean() - np.clip(b, None, b_hi).mean())
ax.hist(bias_sep, bins=60, color=C_BAD, alpha=0.7, label="раздельные трешолды по группам")
ax.hist(bias_common, bins=60, color=C_TRUE, alpha=0.7, label="общий трешолд (по пре-периоду/объединённым)")
ax.axvline(0, color="black", lw=2)
ax.set_xlabel("оценка эффекта в A/A (истина = 0)")
ax.set_title("Раздельные трешолды добавляют шум→смещение оценки;\nобщий порог — обязателен")
ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v41_thresholds_bias.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 42–43. CUPED
# ============================================================
rho = 0.8
pre = rng.normal(0, 1, 1500)
post_test = rho * pre + rng.normal(0, np.sqrt(1 - rho ** 2), 1500)
post_ctrl = rho * pre + rng.normal(0, np.sqrt(1 - rho ** 2), 1500)
theta = np.cov(pre, post_ctrl)[0, 1] / np.var(pre)
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
axes[0].scatter(pre[:300], post_ctrl[:300], s=22, alpha=0.65, color=C_MAIN)
xs = np.linspace(-3.5, 3.5, 50)
axes[0].plot(xs, theta * xs, color=C_BAD, lw=2.6, label=f"θ·pre, θ = {theta:.2f}")
axes[0].set_xlabel("pre-метрика (до эксперимента)"); axes[0].set_ylabel("метрика (в эксперименте)")
axes[0].set_title("CUPED: вычитаем из метрики её «предсказуемую» часть", fontsize=10)
axes[0].legend()
se_raw = post_ctrl.std() / np.sqrt(1500) * np.sqrt(2)
se_cup = (post_ctrl - theta * pre).std() / np.sqrt(1500) * np.sqrt(2)
axes[1].errorbar([0], [0], yerr=[1.96 * se_raw], fmt="o",
                 markersize=11, capsize=7, color=C_BAD, lw=2.6)
axes[1].errorbar([1], [0], yerr=[1.96 * se_cup], fmt="o",
                 markersize=11, capsize=7, color=C_TRUE, lw=2.6)
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["raw", "CUPED"])
axes[1].set_ylabel("95% CI эффекта (ширина)")
axes[1].set_title(f"CI сжимается в 1/√(1−ρ²) = {1/np.sqrt(1-rho**2):.2f} раза\n(ρ={rho}: экономия ~40% наблюдений)", fontsize=10)
fig.suptitle("CUPED: коррекция пре-периодом — та же идея, что парный дизайн из М2",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v42_cuped_scatter_ci.png", bbox_inches="tight"); plt.close(fig)

rhos = np.linspace(0, 0.95, 100)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(rhos, rhos ** 2 * 100, lw=2.6, color=C_MAIN, label="сколько % наблюдений сэкономили")
ax.axvline(0.8, color=C_ACC, ls=":", lw=2)
ax.text(0.81, 40, "ρ=0.8 → экономия 64%", color=C_ACC, fontweight="bold", fontsize=10)
ax.set_xlabel("корреляция пре-метрики с метрикой (ρ)")
ax.set_ylabel("экономия наблюдений, %")
ax.set_title("Экономия независимых наблюдений: ρ²; множитель трафика 1/(1−ρ²)")
ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v43_cuped_savings.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 44. CUPED vs CUPAC (несколько ковариат)
# ============================================================
n = 1200
pre1 = rng.normal(0, 1, n)
pre2 = 0.5 * pre1 + rng.normal(0, 1, n)
pre3 = rng.normal(0, 1, n)
y = 0.55 * pre1 + 0.35 * pre2 + 0.15 * pre3 + rng.normal(0, 0.7, n)
X = np.column_stack([np.ones(n), pre1, pre2, pre3])
# Обучаем прогноз на независимой истории, не на оцениваемых outcomes.
xh1 = rng.normal(0, 1, 2400)
xh2 = .5*xh1 + rng.normal(0, 1, 2400)
xh3 = rng.normal(0, 1, 2400)
yh = .55*xh1 + .35*xh2 + .15*xh3 + rng.normal(0, .7, 2400)
Xh = np.column_stack([np.ones(2400), xh1, xh2, xh3])
pred = X @ np.linalg.lstsq(Xh, yh, rcond=None)[0]
se_raw = y.std() * np.sqrt(2/n)
se_cuped = (y - 0.62 * pre1).std() * np.sqrt(2/n)
se_cupac = (y - pred).std() * np.sqrt(2/n)
fig, ax = plt.subplots(figsize=(8, 4.1))
vals = [se_raw, se_cuped, se_cupac]
ax.bar(range(3), vals, color=[C_BAD, C_MAIN, C_TRUE], width=0.55, alpha=0.9)
ax.set_xticks(range(3)); ax.set_xticklabels(["raw", "CUPED\n(1 ковариата)", "CUPAC/ML\n(3 ковариаты)"])
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, f"SE = {v:.3f}\nCI ±{1.96*v:.3f}", ha="center", fontweight="bold", fontsize=9)
ax.set_ylabel("проектная SE разности, n=1200 на руку")
ax.set_title("Больше информативных пре-ковариат → сильнее сжатие: от CUPED к CUPAC")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v44_cupac.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 45. Peeking: инфляция α и always-valid
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
checks = np.arange(1, 61)
rej_daily = []
for _ in range(3000):
    a = rng.normal(0, 1, 3000); b = rng.normal(0, 1, 3000)
    hit = False
    for k in range(1, 61):
        n_k = 50 * k
        if stats.ttest_ind(a[:n_k], b[:n_k], equal_var=False).pvalue < 0.05:
            hit = True
            break
    rej_daily.append(hit)
axes[0].bar([0, 1], [5, np.mean(rej_daily) * 100], color=[C_TRUE, C_BAD], width=0.5)
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["смотрим один раз\n(корректно)", "смотрим каждый день\n(peeking)"])
for i, v in enumerate([5, np.mean(rej_daily) * 100]):
    axes[0].text(i, v + 0.6, f"{v:.0f}%", ha="center", fontweight="bold")
axes[0].set_ylabel("доля ложных «эффектов» в A/A")
axes[0].set_title("Подглядывание раздувает α")
t = np.arange(1, 61)
alpha_seq, tau = .05, .10
v = 2 / (50*t)
bound = np.sqrt((1+v/tau**2)*(2*np.log(1/alpha_seq)+np.log1p(tau**2/v)))
axes[1].plot(t, bound, lw=2.6, color=C_ACC, label="двусторонняя normal-mixture граница")
axes[1].plot(t, -bound, lw=2.6, color=C_ACC)
z_paths = []
for s in range(12):
    a = rng.normal(0, 1, 3000); b = rng.normal(0, 1, 3000)
    zs = [(b[:50 * k].mean() - a[:50 * k].mean()) / np.sqrt(2 / (50 * k)) for k in t]
    axes[1].plot(t, zs, lw=1.1, alpha=0.75, color=C_MAIN if not np.any(np.abs(zs) >= bound) else C_BAD)
    z_paths.append(zs)
axes[1].axhline(1.96, color="gray", ls="--", lw=1.4)
axes[1].text(2, 2.15, "фиксированный порог z=1,96", fontsize=8.5)
axes[1].set_xlabel("день теста (50 юзеров/день на группу)")
axes[1].set_ylabel("накопленная z-статистика")
axes[1].set_title("mSPRT: уровень контролируется при любом времени остановки")
axes[1].legend(fontsize=8.5, loc="lower right")
fig.suptitle("Два мира: фиксированный горизонт против последовательного теста",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v45_peeking_sequential.png", bbox_inches="tight"); plt.close(fig)

print("Готово: v32–v45 (14 визуализаций) в", OUTPUT_DIR)
