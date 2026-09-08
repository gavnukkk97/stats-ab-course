# %% [markdown]
# # Визуализации для М2 «Проверка гипотез»
# Единый стиль с М1: заголовок = вывод. Запуск: python3 make_visuals_m2.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

rng = np.random.default_rng(202)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_ACC, C_TRUE, C_BAD = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

# ============================================================
# 13. Распределение p-value: под H0 и под H1
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
n, reps = 200, 10000
mu, sd, eff = 1240.0, 300.0, 60.0
p_h0, p_h1 = [], []
for _ in range(reps):
    a = rng.normal(mu, sd, n); b0 = rng.normal(mu, sd, n); b1 = rng.normal(mu + eff, sd, n)
    p_h0.append(stats.ttest_ind(a, b0, equal_var=False).pvalue)
    p_h1.append(stats.ttest_ind(a, b1, equal_var=False).pvalue)
axes[0].hist(p_h0, bins=40, color=C_MAIN, alpha=0.85)
axes[0].set_title("Эффекта нет (H0): p-value равномерен\n«p<0.05» случается ровно в 5% тестов")
axes[0].set_xlabel("p-value"); axes[0].set_ylabel("число тестов")
axes[0].axhline(reps / 40, color=C_BAD, ls="--", lw=1.5)
axes[1].hist(p_h1, bins=40, color=C_ACC, alpha=0.9)
axes[1].set_title(f"Эффект есть (+{eff} ₽ при n={n}): p сыплется к нулю\nдоля p<0.05 = мощность ≈ {np.mean(np.array(p_h1)<0.05):.0%}")
axes[1].set_xlabel("p-value")
fig.suptitle("A/A и A/B глазами p-value: один график — вся логика тестов",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("v13_pvalue_h0_h1.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 14. Матрица ошибок 2×2
# ============================================================
fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
cells = [
    (0.3, 3.2, C_TRUE, "Верно отрицательное", "тест молчит\n«эффекта нет» — и его нет", "(1−α)"),
    (5.2, 3.2, C_BAD,  "Ошибка I рода (FP)", "ложная тревога:\n«эффект есть» — а его нет", "α = 5%"),
    (0.3, 0.3, C_BAD,  "Ошибка II рода (FN)", "пропуск:\n«эффекта нет» — а он есть", "β = 20%"),
    (5.2, 0.3, C_TRUE, "Верно положительное", "тест ловит эффект\n= МОЩНОСТЬ", "1−β = 80%"),
]
for x, y, c, t1, t2, t3 in cells:
    ax.add_patch(plt.Rectangle((x, y), 4.4, 2.5, fc=c, alpha=0.18, ec=c, lw=2.4))
    ax.text(x + 2.2, y + 2.05, t1, ha="center", fontsize=11, fontweight="bold", color=c)
    ax.text(x + 2.2, y + 1.15, t2, ha="center", fontsize=9.5)
    ax.text(x + 2.2, y + 0.42, t3, ha="center", fontsize=10, fontweight="bold")
ax.text(2.5, 5.75, "H0 верна (эффекта нет)", ha="center", fontsize=11)
ax.text(7.4, 5.75, "H1 верна (эффект есть)", ha="center", fontsize=11)
ax.text(0.05, 4.45, "тест не отверг", fontsize=10, rotation=0)
ax.text(0.05, 1.55, "тест отвергает H0", fontsize=10)
ax.set_title("Матрица ошибок: α покупаем заранее, β — дизайном (n и эффект)")
fig.savefig("v14_error_matrix.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 15. Кривые мощности
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.3))
mu, sd, n_per = 1240.0, 300.0, 100
ns = np.arange(50, 3001, 50)
for eff, c in [(30, C_BAD), (60, C_MAIN), (120, C_TRUE)]:
    z_a = 1.96
    power = stats.norm.cdf(np.sqrt(2 * n_per / 2) * eff / sd * np.sqrt(1) / np.sqrt(2) - z_a)
    ns_arr = np.array(ns)
    pw = stats.norm.cdf(eff / (sd * np.sqrt(2 / ns_arr)) - z_a)
    ax.plot(ns_arr, pw, lw=2.4, color=c, label=f"эффект +{eff} ₽")
ax.axhline(0.8, color="gray", ls=":", lw=1.6); ax.text(2600, 0.82, "power 80%", fontsize=9)
ax.axhline(0.05, color="gray", ls=":", lw=1.6); ax.text(2600, 0.07, "α = 5%", fontsize=9)
ax.set_xlabel("n в каждой группе"); ax.set_ylabel("мощность")
ax.set_title("Мощность растёт с n и эффектом: маленький эффект = огромные выборки")
ax.legend()
fig.tight_layout(); fig.savefig("v15_power_curves.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 16. Type M: преувеличение значимых эффектов
# ============================================================
true_eff, n, sd, reps = 30.0, 250, 300.0, 20000
effs = []
for _ in range(reps):
    a = rng.normal(0, sd, n); b = rng.normal(true_eff, sd, n)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    if p < 0.05:
        effs.append(b.mean() - a.mean())
effs = np.array(effs)
fig, ax = plt.subplots(figsize=(8, 4.3))
ax.hist(effs, bins=60, color=C_MAIN, alpha=0.85, label=f"«значимые» эффекты (p<0.05)")
ax.axvline(true_eff, color=C_TRUE, lw=3, label=f"истинный эффект = {true_eff:.0f} ₽")
ax.axvline(effs.mean(), color=C_BAD, lw=2.4, ls="--",
           label=f"среднее значимых = {effs.mean():.0f} ₽ (×{effs.mean()/true_eff:.1f})")
ax.set_xlabel("оценённый эффект, ₽"); ax.set_ylabel("число тестов")
power_true = stats.norm.cdf(true_eff / (sd * np.sqrt(2 / n)) - 1.96)
ax.set_title(f"Type M: при мощности ~{power_true:.0%} значимые оценки в среднем "
             f"×{effs.mean()/true_eff:.1f} больше истины")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig("v16_type_m.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 17. Student vs Уэлч при неравных дисперсиях
# ============================================================
n1, n2, reps = 1000, 100, 4000
mu1 = mu2 = 0.0
res_s, res_w = [], []
for _ in range(reps):
    a = rng.normal(mu1, 1.0, n1)
    b = rng.normal(mu2, 3.0, n2)
    res_s.append(stats.ttest_ind(a, b, equal_var=True).pvalue)
    res_w.append(stats.ttest_ind(a, b, equal_var=False).pvalue)
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.bar([0, 1], [np.mean(np.array(res_s) < 0.05), np.mean(np.array(res_w) < 0.05)],
       color=[C_BAD, C_TRUE], alpha=0.9, width=0.55)
ax.axhline(0.05, color="black", ls="--", lw=1.6)
ax.text(1.32, 0.052, "номинал α=5%", fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(["t Стьюдента (equal_var=True)", "t Уэлча"])
ax.set_ylabel("фактическая доля p<0.05 под H0")
for i, v in enumerate([np.mean(np.array(res_s) < 0.05), np.mean(np.array(res_w) < 0.05)]):
    ax.text(i, v + 0.008, f"{v:.1%}", ha="center", fontweight="bold")
ax.set_title("Неравные дисперсии + разные n: Стьюдент врёт про α, Уэлч держит\n(n₁=1000, σ₁=1; n₂=100, σ₂=3 — эффектов нет)")
fig.tight_layout(); fig.savefig("v17_welch.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 18. Парный vs непарный дизайн
# ============================================================
n = 60
before = rng.normal(70, 12, n)
change = rng.normal(3, 6, n)
after = before + change
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), sharey=True)
for i, (ax, ttl) in enumerate([(axes[0], "Непарный взгляд: два облака"),
                               (axes[1], "Парный взгляд: одна разность")]):
    if i == 0:
        ax.scatter(np.random.uniform(-0.18, 0.18, n), before, s=42, color=C_MAIN, alpha=0.8, label="до")
        ax.scatter(np.random.uniform(0.82, 1.18, n), after, s=42, color=C_ACC, alpha=0.8, label="после")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["до", "после"])
        ax.legend(); sd_ = after.std()
        ax.set_title(f"{ttl}\nSD «после» = {sd_:.1f} → шум глушит сдвиг", fontsize=10)
    else:
        for b_, ch_ in zip(before, change):
            ax.plot([0, 1], [b_, b_ + ch_], color="gray", lw=0.7, alpha=0.5)
        ax.scatter(np.zeros(n), change, s=42, color=C_TRUE, alpha=0.85)
        ax.scatter(np.ones(n), change, s=42, color=C_TRUE, alpha=0.85)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["", ""])
        sd_d = change.std()
        ax.set_title(f"{ttl}\nSD разностей = {sd_d:.1f} — юзер свой собственный контроль", fontsize=10)
    ax.set_ylabel("NPS-скор")
fig.suptitle("Парный дизайн = бесплатное снижение дисперсии (задел на CUPED в М3)",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("v18_paired_design.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 19. Что на самом деле проверяет Манн–Уитни
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
x1 = rng.lognormal(np.log(800), 0.5, 4000)
x2 = rng.lognormal(np.log(860), 0.5, 4000)
axes[0].hist(x1, bins=80, color=C_MAIN, alpha=0.75, density=True, label="A")
axes[0].hist(x2, bins=80, color=C_ACC, alpha=0.75, density=True, label="B (сдвиг)")
axes[0].set_title("Одинаковая форма, сдвиг:\nМанн–Уитни = про медианы/сдвиг, всё честно", fontsize=10)
axes[0].legend()
y1 = rng.lognormal(np.log(800), 0.35, 4000)
y2 = rng.gamma(2.2, 420, 4000)
axes[1].hist(y1, bins=80, color=C_MAIN, alpha=0.75, density=True, label="A")
axes[1].hist(y2, bins=80, color=C_ACC, alpha=0.75, density=True, label="B (другая форма)")
m1, m2 = np.median(y1), np.median(y2)
axes[1].axvline(m1, color=C_MAIN, lw=2.4); axes[1].axvline(m2, color=C_ACC, lw=2.4)
axes[1].set_title(f"Разные формы (медианы {m1:.0f} ≈ {m2:.0f}):\nMW «значим», но это НЕ разница медиан", fontsize=10)
axes[1].legend()
fig.suptitle("Манн–Уитни проверяет стохастическое превосходство P(X>Y), а не медианы",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("v19_mann_whitney.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 20. Схема permutation-теста
# ============================================================
a = rng.normal(10.2, 2, 24)
b = rng.normal(11.4, 2, 24)
pool = np.concatenate([a, b])
obs = b.mean() - a.mean()
perm = np.array([rng.permutation(pool)[24:].mean() - rng.permutation(pool)[:24].mean()
                 for _ in range(4000)])
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(perm, bins=60, color="#999", alpha=0.85, label="перетасовки меток (эффекта нет)")
ax.axvline(obs, color=C_BAD, lw=3, label=f"наблюдаемый эффект = {obs:.2f}")
p_perm = np.mean(np.abs(perm) >= abs(obs))
ax.set_title(f"Permutation-тест: перемешиваем ярлыки групп\nнаблюдаемый эффект экстремален в {p_perm:.1%} миров → p-value = {p_perm:.3f}")
ax.set_xlabel("разность средних после перемешивания")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig("v20_permutation.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 21. Инфляция FWER и «20 метрик»
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
m = np.arange(1, 51)
axes[0].plot(m, 1 - (1 - 0.05) ** m, lw=2.6, color=C_BAD)
axes[0].axhline(0.64, color="gray", ls=":", lw=1.4)
axes[0].annotate("m=20: 64%\nхотя бы одна ложная", xy=(20, 1 - 0.95**20),
                 xytext=(26, 0.45), arrowprops=dict(arrowstyle="->", color=C_BAD),
                 color=C_BAD, fontweight="bold")
axes[0].set_xlabel("число гипотез m"); axes[0].set_ylabel("FWER = 1−(1−0.05)^m")
axes[0].set_title("Чем больше проверок, тем вернее «открытие»")
false_hits = np.zeros(21)
for _ in range(10000):
    false_hits[np.sum(rng.uniform(size=20) < 0.05)] += 1
axes[1].bar(np.arange(21), false_hits / 10000, color=C_MAIN, alpha=0.9)
axes[1].set_title("10000 A/A-миров × 20 метрик: сколько ложных «значимых»")
axes[1].set_xlabel("число метик с p<0.05 (все ложные!)"); axes[1].set_ylabel("доля миров")
axes[1].text(0.98, 0.6, "в среднем 1 метрика «прокрасилась»;\n64% миров — хотя бы одна", transform=axes[1].transAxes,
             ha="right", fontsize=9.5, bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
fig.tight_layout(); fig.savefig("v21_multiple_testing.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 22. Бонферрони vs Холм vs BH: пороги на отсортированных p
# ============================================================
rng2 = np.random.default_rng(7)
p = np.sort(np.concatenate([rng2.uniform(0, 0.02, 6), rng2.uniform(0.1, 0.9, 14)]))
m_ = 20
i = np.arange(1, m_ + 1)
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.scatter(i, p, s=46, color=C_MAIN, zorder=3, label="отсортированные p-value (6 «настоящих» + шум)")
ax.step(i, 0.05 * i / m_, where="post", color=C_BAD, lw=2.2, label="BH: порог 0.05·i/m (FDR)")
ax.axhline(0.05 / m_, color="#8172B3", lw=2.2, label=f"Бонферрони: α/m = {0.05/m_:.4f}")
ax.step(i, 0.05 * m_ / (m_ - i + 1), where="post", color=C_ACC, lw=2.2, ls="--", label="Холм: 0.05·m/(m−i+1)")
ax.set_xlabel("ранг i (отсортированные p)"); ax.set_ylabel("p-value")
ax.set_title("Поправки на одной картине: Бонферрони режет всё, Холм — то же FWER, но мягче, BH — пропускает больше")
ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig("v22_corrections.png", bbox_inches="tight"); plt.close(fig)

print("Готово: 10 визуализаций М2 сохранены в", __file__.rsplit("/", 1)[0])
