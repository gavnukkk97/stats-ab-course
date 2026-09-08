# %% [markdown]
# # Визуализации для М1 «Базовая статистика»
# Единый стиль: заголовок = вывод, подписи объясняют чтение графика.
# Запуск: python3 make_visuals_m1.py  (png сохраняются рядом)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_MED, C_MEAN, C_TRUE, C_BAD = "#4C72B0", "#DD8452", "#DD8452", "#55A868", "#C44E52"

def takeaway(fig, text):
    fig.suptitle(text, fontsize=10, fontweight="bold", color="#333",
                 y=0.005, va="bottom", ha="center")

# ============================================================
# 1. Среднее vs медиана на скошенном распределении
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.2))
data = rng.lognormal(mean=np.log(900), sigma=1.1, size=5000)
ax.hist(data, bins=120, color=C_MAIN, alpha=0.75)
m, med = data.mean(), np.median(data)
ax.axvline(m, color=C_MEAN, lw=2.4)
ax.axvline(med, color="#55A868", lw=2.4)
ax.annotate(f"среднее\n{m:,.0f} ₽".replace(",", " "), xy=(m, ax.get_ylim()[1]*0.8),
            xytext=(m*1.35, ax.get_ylim()[1]*0.86), color=C_MEAN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_MEAN))
ax.annotate(f"медиана\n{med:,.0f} ₽".replace(",", " "), xy=(med, ax.get_ylim()[1]*0.45),
            xytext=(med*0.35, ax.get_ylim()[1]*0.55), color="#2d6a4f", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2d6a4f"))
ax.set_title("Средний чек «ЕдаДома»: среднее в 2.5 раза правее медианы")
ax.set_xlabel("сумма заказа, ₽"); ax.set_ylabel("число заказов")
ax.text(0.98, 0.6, "хвост тянет среднее →\n«средний клиент» не существует",
        transform=ax.transAxes, ha="right", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd"))
fig.tight_layout(); fig.savefig("v01_mean_vs_median.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 2. Эффект кита: с китом и без
# ============================================================
orders = np.array([690, 720, 760, 810, 850, 890, 980, 1240, 1580, 11200])
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, d, ttl in [(axes[0], orders, "Все 10 заказов (с «китом» 11 200 ₽)"),
                   (axes[1], orders[:-1], "Без корпоративного заказа")]:
    ax.scatter(d, np.zeros_like(d), s=140, color=C_MAIN, zorder=3, clip_on=False)
    m, med = d.mean(), np.median(d)
    ax.axvline(m, color=C_MEAN, lw=2.4); ax.axvline(med, color="#55A868", lw=2.4)
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("чек, ₽"); ax.set_yticks([])
    ax.text(m, 0.012, f"mean {m:,.0f}".replace(",", " "), color=C_MEAN,
            ha="center", fontweight="bold")
    ax.text(med, -0.004, f"median {med:,.0f}", color="#2d6a4f",
            ha="center", fontweight="bold", va="top")
    ax.set_ylim(-0.006, 0.014)
axes[0].annotate("кит", xy=(11200, 0), xytext=(9200, 0.008),
                 fontweight="bold", color=C_BAD,
                 arrowprops=dict(arrowstyle="->", color=C_BAD))
takeaway(fig, "Один заказ: среднее −52%, медиана −7% → медиана устойчива к выбросам, среднее — нет")
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig("v02_whale_effect.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 3. Закон малых чисел: траектории выборочного среднего
# ============================================================
true_mean = 1240
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
for ax, n in [(axes[0], 20), (axes[1], 500)]:
    for _ in range(12):
        path = rng.lognormal(np.log(900), 1.1, size=n).cumsum() / np.arange(1, n + 1)
        ax.plot(np.arange(1, n + 1), path, lw=1.2, alpha=0.8)
    ax.axhline(true_mean, color=C_TRUE, lw=2.4)
    ax.text(1, true_mean * 1.12, "истинное среднее", color=C_TRUE, fontweight="bold")
    ax.set_title(f"n = {n}: среднее «бродит»" if n == 20 else f"n = {n}: среднее прижимается",
                 fontsize=10)
    ax.set_xlabel("наблюдение №"); ax.set_ylabel("текущее выборочное среднее")
    ax.set_ylim(0, 4200)
takeaway(fig, "Малая выборка — пьяный шаг: вывод на n=20 может быть каким угодно; n растёт → прижимается к истине")
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig("v03_law_small_numbers.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 4. Зоопарк распределений продуктовых метрик
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
p = 0.12; n = 1000
k = np.arange(0, 260)
axes[0, 0].bar(k, stats.binom.pmf(k, n, p), color=C_MAIN)
axes[0, 0].set_title(f"Биномиальное — конверсия\n{n} визитов, p={p}: в среднем {n*p:.0f} заказов")
axes[0, 0].set_xlabel("число заказов из 1000 визитов")
lam = 2.3
k = np.arange(0, 12)
axes[0, 1].bar(k, stats.poisson.pmf(k, lam), color=C_MAIN)
axes[0, 1].set_title(f"Пуассон — заказы на юзера\nλ={lam}: SD=√λ≈{np.sqrt(lam):.1f} (полка точности)")
axes[0, 1].set_xlabel("заказов за месяц")
x = np.linspace(10, 58, 400)
axes[1, 0].plot(x, stats.norm.pdf(x, 34, 6), color=C_MAIN, lw=2.4)
axes[1, 0].fill_between(x, stats.norm.pdf(x, 34, 6), where=(x > 46), color=C_BAD, alpha=0.5)
axes[1, 0].set_title("Нормальное — время доставки\nμ=34 мин, σ=6: SLA «46 мин» = P(>46)≈2.3%")
axes[1, 0].set_xlabel("минут")
data = rng.lognormal(np.log(900), 1.1, 8000)
axes[1, 1].hist(data, bins=90, color=C_MAIN, alpha=0.8)
axes[1, 1].set_title("Логнормальное — сумма чека\nденьги почти всегда живут так (хвост → вправо)")
axes[1, 1].set_xlabel("₽"); axes[1, 1].set_ylabel("плотность/частота")
for ax in axes.flat: ax.set_ylabel("вероятность" if ax in [axes[0,0], axes[0,1]] else ax.get_ylabel())
fig.suptitle("Четыре распределения — четыре продуктовые метрики", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig("v04_distribution_zoo.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 5. Выбор числа бинов гистограммы
# ============================================================
data = rng.lognormal(np.log(900), 1.1, size=5000)
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2), sharey=True)
for ax, bins, name, verdict in [
    (axes[0], 5, "5 бинов: «слишком грубо»", "спрячет структуру"),
    (axes[1], 14, "14 бинов (Стерджесс)", "ок для быстрого взгляда"),
    (axes[2], 50, "50 бинов (Скотт)", "рабочий вариант"),
    (axes[3], 200, "200 бинов: «шум»", "пиками управляет случай")]:
    ax.hist(data, bins=bins, color=C_MAIN, alpha=0.85)
    ax.set_title(f"{name}", fontsize=9.5)
    ax.text(0.97, 0.62, verdict, transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
    ax.set_xlabel("чек, ₽")
axes[0].set_ylabel("частота")
fig.suptitle("Одни данные — четыре истории: число бинов меняет вывод",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("v05_bins_choice.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 6. Как читать ECDF
# ============================================================
times = rng.gamma(6, 5.6, 4000)  # время доставки, мин
fig, axes = plt.subplots(2, 1, figsize=(8, 6.2))
axes[0].hist(times, bins=60, color=C_MAIN, alpha=0.85)
axes[0].set_title("Время доставки: гистограмма (отвечает «сколько где»)")
axes[0].set_xlabel("минут")
xs = np.sort(times); ys = np.arange(1, len(xs) + 1) / len(xs)
axes[1].plot(xs, ys, color=C_MAIN, lw=2.4)
for q, c, lab in [(50, "#55A868", "P50 (медиана)"), (90, C_BAD, "P90")]:
    v = np.percentile(times, q)
    axes[1].axvline(v, color=c, lw=2, ls="--")
    axes[1].axhline(q / 100, color=c, lw=1, alpha=0.6)
    axes[1].annotate(f"{lab} = {v:.0f} мин", xy=(v, q / 100), xytext=(v + 6, q / 100 - 0.13),
                     color=c, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=c))
axes[1].set_title("ECDF (отвечает «какая доля ≤ X») — SLA читаются отсюда напрямую")
axes[1].set_xlabel("минут"); axes[1].set_ylabel("доля заказов ≤ X")
axes[1].text(np.percentile(times, 97), 0.35, "читаем P90:\nвертикаль ↑ до кривой,\nгоризонталь → ось X",
             fontsize=9, ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
fig.tight_layout(); fig.savefig("v06_ecdf_reading.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 7. QQ-plot: сырые деньги vs логарифм
# ============================================================
data = rng.lognormal(np.log(900), 1.1, 3000)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
for ax, d, ttl in [(axes[0], data, "Сырой чек: точки улетают от прямой"),
                   (axes[1], np.log(data), "ln(чек): легли на прямую → логнормала")]:
    stats.probplot(d, dist="norm", plot=ax)
    ax.get_lines()[0].set(marker="o", markersize=2.5, color=C_MAIN, alpha=0.6)
    ax.get_lines()[1].set(color=C_BAD, lw=2)
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("теоретические квантили N(0,1)")
fig.suptitle("QQ-plot: экспресс-тест «нормальные ли данные»", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("v07_qq_plot.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 8. ЦПТ: от скошенной совокупности к нормальным средним
# ============================================================
fig = plt.figure(figsize=(10, 5.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.4])
ax0 = fig.add_subplot(gs[0, :])
pop = rng.lognormal(np.log(900), 1.1, 200000)
ax0.hist(pop, bins=200, color="#888", alpha=0.8)
ax0.set_title("Совокупность (чек): скошена и с хвостом — «ненормальная»", fontsize=10)
ax0.set_xlim(0, 8000)
for i, n_ in enumerate([5, 30, 1000]):
    ax = fig.add_subplot(gs[1, i])
    means = rng.lognormal(np.log(900), 1.1, size=(20000, n_)).mean(axis=1)
    ax.hist(means, bins=80, density=True, color=C_MAIN, alpha=0.8)
    xs = np.linspace(means.min(), means.max(), 300)
    ax.plot(xs, stats.norm.pdf(xs, means.mean(), means.std()), color=C_BAD, lw=2.2)
    ax.set_title(f"средние при n={n_}\n(красное — нормальная кривая)", fontsize=9)
    if n_ == 5:
        ax.text(0.97, 0.6, "само ещё косое", transform=ax.transAxes, ha="right",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
    if n_ == 1000:
        ax.text(0.97, 0.6, "почти идеал", transform=ax.transAxes, ha="right",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#d8f3dc"))
fig.suptitle("ЦПТ: данные — любые, средние — нормальные (чем больше n, тем ближе)",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("v08_clt_intuition.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 9. SE ~ 1/√n: цена точности
# ============================================================
ns = np.array([10, 50, 100, 500, 1000, 5000])
sigma = 2350
se = sigma / np.sqrt(ns)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.loglog(ns, se, "o-", color=C_MAIN, lw=2.4, ms=8)
for n_, s_ in zip(ns, se):
    ax.annotate(f"n={n_}\n±{s_:,.0f} ₽".replace(",", " "), xy=(n_, s_),
                xytext=(n_ * 1.15, s_ * 1.15), fontsize=9)
ax.set_xlabel("размер выборки n (log)"); ax.set_ylabel("SE среднего (log)")
ax.set_title("Точность дорожает: ×4 данных → только ×2 точнее")
ax.text(0.05, 0.15, "наклон = −1/2 (закон √n)", transform=ax.transAxes,
        fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd"))
fig.tight_layout(); fig.savefig("v09_se_sqrt_n.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 10. Покрытие: 100 доверительных интервалов
# ============================================================
n, mu, sigma = 100, 1240.0, 2350.0
means = rng.normal(mu, sigma / np.sqrt(n), size=100)
hw = 1.96 * sigma / np.sqrt(n)
fig, ax = plt.subplots(figsize=(8.5, 5.5))
covered = 0
for i, m_ in enumerate(means):
    ok = abs(m_ - mu) <= hw
    covered += ok
    ax.plot([m_ - hw, m_ + hw], [i, i], color=C_TRUE if ok else C_BAD,
            lw=2.4, alpha=0.9 if ok else 1.0)
ax.axvline(mu, color="black", lw=2, ls="--")
ax.text(mu + 12, 104, "истинное среднее", fontsize=10)
ax.set_ylim(-2, 110); ax.set_ylabel("номер эксперимента")
ax.set_xlabel("95% доверительный интервал")
ax.set_title(f"100 экспериментов по n={n}: {covered} интервалов накрыли истину, "
             f"{100 - covered} — промахнулись (красные)")
fig.tight_layout(); fig.savefig("v10_ci_coverage.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 11. Схема бутстрапа
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
sample = rng.choice(np.concatenate([rng.lognormal(np.log(900), 1.1, 40)]), 40)
axes[0].scatter(sample, rng.uniform(-1, 1, 40), s=90, color=C_MAIN, clip_on=False)
axes[0].set_title("1) Есть одна выборка\n(суррогат совокупности)", fontsize=10)
axes[0].set_yticks([]); axes[0].set_xlabel("значения")
for b, dy in [(1, 0.55), (2, 0.0), (3, -0.55)]:
    boot = rng.choice(sample, 40, replace=True)
    axes[1].scatter(boot, dy + rng.uniform(-0.18, 0.18, 40), s=38, color=C_MED, alpha=0.85)
axes[1].set_title("2) Тысячи повторных выборок\n(с возвратом, объёмом n)", fontsize=10)
axes[1].set_yticks([]); axes[1].set_xlabel("значения")
axes[1].text(0.5, 0.92, "★ каждый раз — своя медиана/перцентиль", transform=axes[1].transAxes,
             ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
meds = [np.median(rng.choice(sample, 40, replace=True)) for _ in range(4000)]
axes[2].hist(meds, bins=60, color=C_MAIN, alpha=0.85)
lo, hi = np.percentile(meds, [2.5, 97.5])
axes[2].axvline(lo, color=C_BAD, ls="--", lw=2); axes[2].axvline(hi, color=C_BAD, ls="--", lw=2)
axes[2].set_title("3) Распределение статистики\nи её 95% интервал", fontsize=10)
axes[2].set_xlabel("медиана бут-выборки")
fig.suptitle("Бутстрап: CI для любой статистики без формул", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("v11_bootstrap_schema.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 12. Ловушка объёма бут-выборки
# ============================================================
n = 200
sample = rng.lognormal(np.log(900), 1.1, n)
fig, ax = plt.subplots(figsize=(8, 4.2))
labels, widths, xs_plot = [], [], []
for i, m_ in enumerate([n // 5, n, 5 * n]):
    meds = [np.median(rng.choice(sample, m_, replace=True)) for _ in range(3000)]
    lo, hi = np.percentile(meds, [2.5, 97.5])
    labels.append(f"бут-выборка {m_}"); widths.append(hi - lo); xs_plot.append(hi - lo)
bars = ax.bar(range(3), widths, color=[C_BAD, C_MAIN, C_BAD], alpha=0.85)
ax.set_xticks(range(3)); ax.set_xticklabels(labels)
for i, (w, b) in enumerate(zip(widths, bars)):
    ax.text(i, w + 4, f"ширина {w:.0f} ₽", ha="center", fontweight="bold")
ax.set_ylabel("ширина 95% CI медианы")
ax.set_title("Правило нарушено → интервал врёт: меньше n — ложно широкий, больше n — ложно узкий")
ax.text(0.02, 0.9, "правильно: бут-выборка ровно объёма n", transform=ax.transAxes,
        fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc="#d8f3dc"))
fig.tight_layout(); fig.savefig("v12_bootstrap_trap.png", bbox_inches="tight"); plt.close(fig)

print("Готово: 12 визуализаций сохранены в", __file__.rsplit("/", 1)[0])
