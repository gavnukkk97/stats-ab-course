from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "course" / "modules" / "M2" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Визуализации М2 — вторая партия (под VIS-маркеры уроков)
# Запуск: python3 scripts/figures/make_visuals_m2b.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
sys.path.insert(0, str(ROOT / "course" / "lib"))
from foundations import normal_power, coin_pvalues, permutation_differences, permutation_pvalue, holm_thresholds, type_m, mc_interval, mc_report

rng = np.random.default_rng(777)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_ACC, C_TRUE, C_BAD = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

# ============================================================
# 23. Шкала-линейка p-value для монеты (10 бросков)
# ============================================================
ks = np.arange(11)
p_two = coin_pvalues(10)
fig, ax = plt.subplots(figsize=(9, 3.4))
cols = [C_BAD if pv < 0.05 else C_MAIN for pv in p_two]
ax.bar(ks, p_two * 100, color=cols, alpha=0.9)
for k, pv in zip(ks, p_two):
    ax.text(k, pv * 100 + 1.6, f"{pv*100:.0f}%" if pv * 100 >= 1 else f"{pv*100:.1f}%",
            ha="center", fontsize=9, fontweight="bold" if pv < 0.05 else "normal")
ax.axhline(5, color="black", ls="--", lw=1.4)
ax.text(10.3, 6, "α = 5%", fontsize=9)
ax.set_xticks(ks); ax.set_xlabel("сколько орлов из 10 (норма — 5)")
ax.set_ylabel("двусторонний p-value, %")
ax.set_ylim(0, 140)
ax.set_title("Точная проверка честной монеты: p-value учитывает\nнаблюдаемый исход и более экстремальные исходы под H0")
ax.text(0.50, 0.83, "Двусторонняя проверка: 7 орлов → p=34,4%\n8 → 10,9% · 9 → 2,15% · 10 → 0,195%",
        transform=ax.transAxes, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v23_coin_pvalue_scale.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 24. От z к t: плотности
# ============================================================
x = np.linspace(-5, 5, 500)
fig, ax = plt.subplots(figsize=(8, 4.1))
ax.plot(x, stats.norm.pdf(x), color="black", lw=2.6, label="N(0,1): z-тест (σ известна)")
ax.plot(x, stats.t.pdf(x, 3), color=C_BAD, lw=2.2, label="t(3): n=4")
ax.plot(x, stats.t.pdf(x, 20), color=C_ACC, lw=2.2, label="t(20): n=21")
ax.set_ylim(0, 0.43)
for df, c in [(3, C_BAD), (20, C_ACC)]:
    crit = stats.t.ppf(0.975, df)
    ax.axvline(crit, color=c, ls=":", lw=1.6)
    ax.axvline(-crit, color=c, ls=":", lw=1.6)
ax.axvline(1.96, color="black", ls=":", lw=1.6)
ax.text(2.05, 0.40, "критические 2,5%:\n1,96 | t₂₀=2,09 | t₃=3,18", fontsize=9)
ax.set_title("Почему t вместо z: σ оценена → хвосты тяжелее → порог строже")
ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v24_z_to_t.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 25. Скошенный чек: эмпирическая α от n
# ============================================================
def alpha_curve(n_arr, reps=3000):
    res = []
    for n in n_arr:
        rej = 0
        for _ in range(reps):
            a = rng.lognormal(np.log(700), 1.2, n)
            b = rng.lognormal(np.log(700), 1.2, n)
            if stats.ttest_ind(a, b, equal_var=False).pvalue < 0.05:
                rej += 1
        res.append(rej / reps)
    return np.array(res)

ns = np.array([30, 100, 300, 1000, 3000, 10000])
al = alpha_curve(ns, reps=1200)
mc_bounds = np.array([mc_interval(round(rate * 1200), 1200) for rate in al])
fig, ax = plt.subplots(figsize=(8, 4.1))
ax.errorbar(ns, al * 100, yerr=100 * np.array([al - mc_bounds[:, 0], mc_bounds[:, 1] - al]),
            fmt="o-", capsize=3, color=C_BAD, lw=2.4, label="Уэлч, A/A; 95% Monte Carlo CI (1200 миров)")
ax.axhline(5, color="black", ls="--", lw=1.6, label="номинал α = 5%")
ax.set_xscale("log")
ax.axvline(12500, color=C_ACC, ls=":", lw=2)
ax.text(13000, 6.5, "эвристика Денга\n100·s² ≈ 12 500", color=C_ACC, fontsize=9)
for n_, a_ in zip(ns, al):
    ax.text(n_, a_ * 100 + 0.5, f"{a_*100:.1f}%", ha="center", fontsize=9)
ax.set_xlabel("n на группу (log)"); ax.set_ylabel("фактическая доля p<0.05, %")
ax.set_title("Двухвыборочный Уэлч в этой логнормальной модели близок к α = 5%\nДля иных хвостов, статистик и дизайнов калибровку проверяют отдельно")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v25_skew_alpha_n.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 26. Peeking: траектории p-value в A/A
# ============================================================
fig, ax = plt.subplots(figsize=(8.6, 4.4))
mu, sd = 1240.0, 300.0
n_days, day_n = 14, 60
crossed = 0
for s in range(30):
    a = rng.normal(mu, sd, n_days * day_n)
    b = rng.normal(mu, sd, n_days * day_n)
    p = [stats.ttest_ind(a[:(d + 1) * day_n], b[:(d + 1) * day_n], equal_var=False).pvalue
         for d in range(n_days)]
    hit = any(np.array(p[:7]) < 0.05)
    crossed += hit
    ax.plot(range(1, n_days + 1), p, lw=1.2,
            color=C_BAD if hit else "#88a", alpha=0.95 if hit else 0.55)
ax.axhline(0.05, color="black", ls="--", lw=1.8)
ax.text(1, 0.065, "p = 0,05", fontsize=10)
ax.set_yscale("log"); ax.set_ylim(0.0005, 1)
ax.set_xlabel("день теста (каждый день +60 юзеров в группу)")
ax.set_ylabel("p-value (log)")
ax.set_title(f"Подглядывание в A/A-тесте: {crossed}/30 «зелёных» траекторий прокрасились в первую неделю\n(эффекта нет вообще)")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v26_peeking.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 27. ANOVA: разложение вариативности
# ============================================================
means = [42, 45, 44, 50]
data = [rng.normal(m, 3, 12) for m in means]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
grand = np.mean(np.concatenate(data))
ax.axhline(grand, color="black", lw=2, ls="--")
ax.text(3.62, grand + 0.4, "общее среднее", fontsize=9)
for i, d in enumerate(data):
    ax.scatter(np.full(12, i) + rng.uniform(-0.13, 0.13, 12), d, s=42, color=C_MAIN, alpha=0.85)
    m_ = np.mean(d)
    ax.plot([i - 0.25, i + 0.25], [m_, m_], color=C_BAD, lw=3)
    ax.annotate("", xy=(i, m_), xytext=(i, grand),
                arrowprops=dict(arrowstyle="<->", color=C_ACC, lw=1.8))
ax.annotate("SS_within — шум внутри групп\n(разлёт точек вокруг своих средних)", xy=(0.55, 38.5),
            fontsize=9.5, color=C_MAIN,
            bbox=dict(boxstyle="round,pad=0.3", fc="#e8f0fe"))
ax.annotate("SS_between — различия средних\n(стрелки от общего среднего)", xy=(2.35, 52.5),
            fontsize=9.5, color=C_ACC,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
ax.set_xticks(range(4)); ax.set_xticklabels(["стандарт", "экспресс", "pickup", "подписка"])
ax.set_ylabel("дни до повторного заказа")
ax.set_title("ANOVA: F = (шум между группами / шум внутри групп) — во сколько раз различия больше случайности")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v27_anova_ss.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 28. F-распределение с наблюдаемым F
# ============================================================
x = np.linspace(0, 10, 400)
fig, ax = plt.subplots(figsize=(8, 4.0))
ax.plot(x, stats.f.pdf(x, 3, 16), color=C_MAIN, lw=2.6)
crit = stats.f.ppf(0.95, 3, 16)
ax.fill_between(x, stats.f.pdf(x, 3, 16), where=x > crit, color=C_BAD, alpha=0.4)
ax.axvline(crit, color=C_BAD, ls="--", lw=1.8)
ax.axvline(6.46, color=C_ACC, lw=3)
ax.text(6.62, 0.30, "наш F = 6,46\np = 0,0045", color=C_ACC, fontweight="bold", fontsize=10)
ax.text(crit + 0.12, 0.60, f"критический F = {crit:.2f}", color=C_BAD, fontsize=9)
ax.text(0.7, 0.55, "под H0 F ≈ 1\n(различий нет — шум равен шуму)", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
ax.set_xlabel("F-статистика"); ax.set_ylabel("плотность F(3, 16)")
ax.set_title("Наш F глубоко в хвосте: сдвиг средних больше, чем объясняет шум")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v28_anova_f.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 29. Ранговая трансформация
# ============================================================
a = rng.lognormal(np.log(800), 0.9, 120)
b = rng.lognormal(np.log(900), 0.9, 120)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
axes[0].scatter(a, np.zeros(120), s=26, color=C_MAIN, alpha=0.7, label="контроль", clip_on=False)
axes[0].scatter(b, np.ones(120), s=26, color=C_ACC, alpha=0.7, label="тест", clip_on=False)
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["контроль", "тест"])
axes[0].set_xlabel("чек, ₽ (сырые рубли)")
axes[0].set_title("Сырые данные: киты разрывают шкалу — дисперсия огромна", fontsize=10)
axes[0].legend(loc="lower right")
allv = np.concatenate([a, b])
ranks = np.argsort(np.argsort(allv)) + 1
ra, rb = ranks[:120], ranks[120:]
axes[1].scatter(ra, np.zeros(120), s=26, color=C_MAIN, alpha=0.7, clip_on=False)
axes[1].scatter(rb, np.ones(120), s=26, color=C_ACC, alpha=0.7, clip_on=False)
axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["контроль", "тест"])
axes[1].set_xlabel("ранг от 1 до 240")
axes[1].set_title("Те же данные как ранги: ровная шкала, кит — просто «самый большой»", fontsize=10)
fig.suptitle("Ранговая трансформация — сердце Манна–Уитни", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v29_rank_transform.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 30. Киты ломают t-тест: уровень и мощность
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
rej_eq = rej_uneq = rej_un = 0
reps = 2500
for _ in range(reps):
    a = rng.lognormal(np.log(1200) - 1.0**2/2, 1.0, 40)
    b_eq = rng.lognormal(np.log(1200) - 1.0**2/2, 1.0, 40)
    b_un = rng.lognormal(np.log(1200) - 2.0**2/2, 2.0, 40)
    if stats.ttest_ind(a, b_eq, equal_var=False).pvalue < 0.05: rej_eq += 1
    if stats.ttest_ind(a, b_un, equal_var=False).pvalue < 0.05: rej_un += 1
axes[0].bar([0, 1], [rej_eq / reps * 100, rej_un / reps * 100], color=[C_TRUE, C_BAD], width=0.55)
low, high = mc_interval(np.array([rej_eq, rej_un]), reps)
axes[0].errorbar([0, 1], np.array([rej_eq, rej_un])/reps*100, yerr=[(np.array([rej_eq, rej_un])/reps-low)*100, (high-np.array([rej_eq, rej_un])/reps)*100], fmt="none", color="black", capsize=4)
axes[0].axhline(5, color="black", ls="--", lw=1.5)
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["обе группы обычные\n(σ_log=1)", "тест с китами\n(σ_log=2), n=40"])
axes[0].set_ylabel("доля p<0.05 при равных E[X], %")
axes[0].set_title("Уровень при H0: E[A]=E[B]=1200 ₽\nразные хвосты, n=40; черта — α=5%", fontsize=9.5)
for i, v in enumerate([rej_eq / reps * 100, rej_un / reps * 100]):
    axes[0].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")
n_ = 200
sd1 = 1200 * np.sqrt(np.exp(1.0) - 1)
sd2 = 1200 * np.sqrt(np.exp(4.0) - 1)
ns_arr = np.arange(50, 3000, 50)
pw_no = normal_power(30, sd1 * np.sqrt(2 / ns_arr))
pw_wh = normal_power(30, np.sqrt((sd1**2 + sd2**2) / ns_arr))
axes[1].plot(ns_arr, pw_no, color=C_TRUE, lw=2.4, label="без китов")
axes[1].plot(ns_arr, pw_wh, color=C_BAD, lw=2.4, label=f"тяжёлый хвост в тесте (SD ×{sd2/sd1:.1f})")
axes[1].axhline(0.8, color="gray", ls=":", lw=1.3)
axes[1].set_xlabel("n на группу"); axes[1].set_ylabel("normal-аппроксимация мощности (+30 ₽)")
axes[1].set_title("Мощность: раздутая китом дисперсия съедает чувствительность\nнужен кратно больший n", fontsize=9.5)
axes[1].legend(fontsize=9)
fig.suptitle("Тяжёлые хвосты: калибровка уровня и планирование мощности", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v30_whales_break_ttest.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 31. ANOVA как регрессия → CUPED (дизайн-матрица)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
m1 = "y = [ 1 0 0 0 ]  ·  β₀\n    [ 1 1 0 0 ]  ·  β_B\n    [ 1 0 1 0 ]  ·  β_C\n    [ 1 0 0 1 ]  ·  β_D"
axes[0].text(5, 7.6, "ANOVA = регрессия с dummy", ha="center", fontsize=12, fontweight="bold")
axes[0].text(0.4, 2.2, m1, fontsize=11, family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", fc="#e8f0fe"))
axes[0].text(5, 1.0, "коэффициенты = сдвиги групп,\nF-тест = «нужны ли dummy вообще»", ha="center", fontsize=9.5)
axes[1].text(5, 7.6, "…добавили пре-период → ANCOVA/CUPED", ha="center", fontsize=12, fontweight="bold")
m2 = "y = β₀ + β_D·D + θ·ПREFIT\n\nX = [ 1  0  812 ]\n    [ 1  1  940 ]\n    [ 1  0  1055 ]\n    [ 1  1  873 ]"
axes[1].text(0.4, 2.2, m2, fontsize=11, family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", fc="#fff3cd"))
axes[1].text(5, 1.0, "SE эффекта сжимается в 1/√(1−ρ²) раз:\nпри ρ=0.8 — в 1,67 раза короче CI", ha="center", fontsize=9.5,
             color=C_ACC, fontweight="bold")
fig.suptitle("Одна картинка — мост из М2 в М3: регрессионная оптика готовит CUPED",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v31_anova_cuped.png", bbox_inches="tight"); plt.close(fig)

print("Готово: v23–v31 (9 визуализаций) сохранены в", OUTPUT_DIR)
