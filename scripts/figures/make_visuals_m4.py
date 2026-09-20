from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "course" / "modules" / "M4" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Визуализации для М4 «АБ-тесты на практике»
# Единый стиль: заголовок = вывод. Запуск: python3 scripts/figures/make_visuals_m4.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

rng = np.random.default_rng(404)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_ACC, C_TRUE, C_BAD = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

# ============================================================
# 46. Пирамида метрик: OEC, guardrails, инварианты
# ============================================================
fig, ax = plt.subplots(figsize=(8.6, 4.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
rows = [
    (1.5, 6.7, 7.0, C_TRUE, "OEC (целевая)", "драйвер ценности: чувствительная и близкая к деньгам — по ней решение"),
    (0.7, 4.5, 8.6, C_ACC, "Guardrails (ограничения)", "для каждой метрики задаём направление вреда и допустимую границу"),
    (0.2, 2.5, 9.6, "#8C8C8C", "Инварианты (здоровье)", "назначение и pre-признаки; отклонения оцениваем с учётом случайности"),
]
for x, y, w, c, t1, t2 in rows:
    ax.add_patch(plt.Rectangle((x, y), w, 1.7, fc=c, alpha=0.18, ec=c, lw=2.2))
    ax.text(x + w / 2, y + 1.12, t1, ha="center", fontsize=11.5, fontweight="bold", color=c)
    ax.text(x + w / 2, y + 0.45, t2, ha="center", fontsize=9)
ax.annotate("", xy=(5, 6.55), xytext=(5, 6.25),
            arrowprops=dict(arrowstyle="-", color="gray"))
ax.text(5, 8.9, "Пирамида метрик эксперимента", ha="center", fontsize=13, fontweight="bold")
ax.text(5, 8.35, "решение по вершине, тревога по середине, здоровье по низу", ha="center", fontsize=10, color="#444")
ax.text(5, 1.6, "Secondary-метрики (информационные): смотрим с поправкой, не решаем", ha="center",
        fontsize=9.5, style="italic", color="#666")
fig.savefig(OUTPUT_DIR / "v46_oec_pyramid.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 47. Decision rule 2×2
# ============================================================
fig, ax = plt.subplots(figsize=(8.4, 4.9))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
cells = [
    (0.3, 3.1, "#9CC5E8", "Зелёный: эффект > 0, значимо", "ПРОВЕРИТЬ ПОЛЬЗУ", "бизнес-порог + безопасность guardrails"),
    (5.3, 3.1, "#F2B8B5", "Красный: эффект < 0, значимо", "ОТКАТ + разбирательство", "guardrail или гипотеза не та"),
    (0.3, 0.3, "#EDE7DC", "Серый: эффект > 0, не значимо", "нет решения", "оценка + CI; новый протокол\nдля следующей проверки"),
    (5.3, 0.3, "#EDE7DC", "Серый: эффект < 0, не значимо", "нет решения", "CI: что ещё совместимо с данными?"),
]
for x, y, c, t1, t2, t3 in cells:
    ax.add_patch(plt.Rectangle((x, y), 4.4, 2.5, fc=c, alpha=0.75, ec="white", lw=3))
    ax.text(x + 2.2, y + 2.0, t1, ha="center", fontsize=10.5, fontweight="bold")
    ax.text(x + 2.2, y + 1.2, t2, ha="center", fontsize=11, color="#333")
    ax.text(x + 2.2, y + 0.55, t3, ha="center", fontsize=8.5, color="#555")
ax.text(2.5, 5.8, "эффект положительный", ha="center", fontsize=10.5)
ax.text(7.5, 5.8, "эффект отрицательный", ha="center", fontsize=10.5)
ax.text(0.05, 4.35, "значимо", fontsize=10, rotation=90, va="center")
ax.text(0.05, 1.55, "не значимо", fontsize=10, rotation=90, va="center")
ax.set_title("Знак и неопределённость OEC: затем бизнес-порог и безопасность")
fig.savefig(OUTPUT_DIR / "v47_decision_rule.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 48. Сплит-система: слои и слоты
# ============================================================
fig, ax = plt.subplots(figsize=(9, 4.7))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.text(5, 5.7, "Слои = независимые лотереи: одна соль на слой", ha="center", fontsize=11.5, fontweight="bold")
layers = [
    (4.0, "слой 1: поиск", "соль S1", ["тест А: A/B", "свободно"]),
    (2.3, "слой 2: доставка", "соль S2", ["тест B: A/B", "тест C: A/B", "свободно"]),
    (0.6, "слой 3: промо", "соль S3", ["тест D: A/B", "свободно"]),
]
for y, name, salt, arms in layers:
    ax.add_patch(plt.Rectangle((0.3, y), 9.4, 1.5, fc="#F5F5F5", ec="#999", lw=1.5))
    ax.text(0.55, y + 1.12, name, fontsize=10, fontweight="bold")
    ax.text(0.55, y + 0.35, f"hash(user_id + {salt}) → слот", fontsize=8.5, color="#666")
    x = 3.6
    for arm in arms:
        w = 1.5 if arm != "контроль" else 1.5
        c = "#C44E52" if "контроль" in arm else C_MAIN
        ax.add_patch(plt.Rectangle((x, y + 0.25), w, 1.0, fc=c, alpha=0.25, ec=c, lw=1.8))
        ax.text(x + w / 2, y + 0.75, arm, ha="center", fontsize=9)
        x += w + 0.35
ax.text(5, 0.1, "между слоями возможно совместное участие;\nвнутри слоя слоты не пересекаются",
        ha="center", fontsize=9, style="italic", color="#444",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
fig.savefig(OUTPUT_DIR / "v48_split_layers.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 49. SRM: какие отклонения ловит хи-квадрат
# ============================================================
cases = [(49500, 50500), (49995, 50005), (49800, 50200), (495000, 505000)]
fig, ax = plt.subplots(figsize=(9, 4.2))
labels, pvals = [], []
for a, b in cases:
    n = a + b
    chi2 = (a - n / 2) ** 2 / (n / 2) + (b - n / 2) ** 2 / (n / 2)
    p = stats.chi2.sf(chi2, df=1)
    labels.append(f"{a//1000}к/{b//1000}к" if a >= 100000 else f"{a}/{b}")
    pvals.append(max(p, 1e-30))
colors = [C_BAD if p < 0.001 else C_TRUE for p in pvals]
ax.bar(range(len(cases)), [max(p, 1e-25) for p in pvals], color=colors, width=0.55, alpha=0.9)
ax.set_yscale("log"); ax.set_ylim(1e-25, 1e3); ax.axhline(0.001, color="black", ls="--", lw=1.8)
ax.text(len(cases) - 0.4, 0.0016, "порог SRM: p < 0.001", fontsize=9.5, ha="right")
ax.set_xticks(range(len(cases))); ax.set_xticklabels(labels)
for i, p in enumerate(pvals):
    verdict = "SRM!" if p < 0.001 else "нет сигнала"
    ax.text(i, max(p, 1e-25) * 1.3, f"p={p:.1e}\n{verdict}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("p-value хи-квадрат (log)")
ax.set_title("SRM: величина перекоса и размер выборки совместно определяют p-value")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v49_srm.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 50. A/A-тесты: здоровье платформы
# ============================================================
sample_a = rng.normal(size=(2000,80))
sample_b = rng.normal(size=(2000,80))
pvals = stats.ttest_ind(sample_a,sample_b,axis=1,equal_var=False).pvalue
sample_b[:140] += .5
pvals_srm = stats.ttest_ind(sample_a,sample_b,axis=1,equal_var=False).pvalue
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
axes[0].hist(pvals, bins=40, color=C_TRUE, alpha=0.85)
axes[0].set_title("Учебная модель: 2000 сравнений при H0\np-value равномерен (доля <0.05 ≈ 5%)")
axes[0].set_xlabel("p-value")
axes[1].hist(pvals_srm, bins=40, color=C_BAD, alpha=0.85)
axes[1].set_title("В 140 мирах задан сдвиг: это уже не H0\nраспределение p меняется; это не аудит реальной платформы")
axes[1].set_xlabel("p-value")
fig.suptitle("A/A-тест — ЭКГ платформы экспериментов", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v50_aa_health.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 51. Интерференция: switchback и гео
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
for ax, ttl in [(axes[0], "Гео-рандомизация"), (axes[1], "Switchback (время = кластер)")]:
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title(ttl, fontsize=11)
grid = [(1 + 2.6 * i, 1.2 + 1.7 * j) for i in range(3) for j in range(3)]
cols = [C_MAIN, C_ACC]
for i, (x, y) in enumerate(grid):
    axes[0].add_patch(plt.Rectangle((x, y), 2.1, 1.3, fc=cols[i % 2], alpha=0.35, ec=cols[i % 2], lw=2))
    axes[0].text(x + 1.05, y + 0.65, "город", ha="center", fontsize=8)
axes[0].text(5, 0.35, "предполагаем слабые связи между городами; проверяем их", ha="center", fontsize=9, style="italic")
t = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб"]
for i, d in enumerate(t):
    x = 0.7 + i * 1.55
    c = cols[[1,0,1,0,0,1][i]]
    axes[1].add_patch(plt.Rectangle((x, 2.0), 1.35, 2.2, fc=c, alpha=0.35, ec=c, lw=2))
    axes[1].text(x + 0.67, 4.4, d, ha="center", fontsize=9)
    axes[1].text(x + 0.67, 3.1, "T" if [1,0,1,0,0,1][i] else "C", ha="center", fontsize=12, fontweight="bold")
axes[1].text(5, 1.2, "пример случайного назначения временных блоков;\ncarryover и корреляция требуют отдельного учёта", ha="center", fontsize=9, style="italic")
fig.suptitle("Интерференция: возможные дизайны зависят от механизма взаимодействия", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v51_interference_designs.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 52. Эффект новизны
# ============================================================
weeks = np.arange(1, 9)
novelty = 6.5 * np.exp(-0.45 * weeks) + 1.0
fig, ax = plt.subplots(figsize=(8, 4.1))
ax.plot(weeks, novelty, "o-", lw=2.6, color=C_MAIN, ms=8, label="эффект по неделям")
ax.axhline(1.0, color=C_TRUE, ls="--", lw=2, label="долгосрочный эффект")
ax.axvspan(1, 2, color=C_BAD, alpha=0.12)
ax.text(1.06, 5.6, f"первые 2 недели: средний эффект\nв {novelty[:2].mean():.1f} раза выше предела модели", fontsize=9.5, color=C_BAD, fontweight="bold")
ax.set_xlabel("неделя после раскатки"); ax.set_ylabel("эффект, %")
ax.set_title("Учебная модель новизны: короткое и длительное окна измеряют разные эффекты")
ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v52_novelty.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 53. Парадокс Симпсона
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
# Контроль A и тест B имеют разный состав. Внутри обоих сегментов B хуже.
weights_a = np.array([.2, .8]); weights_b = np.array([.8, .2])
cr_a = np.array([.12, .08]); cr_b = np.array([.115, .075])
values = [weights_a @ cr_a * 100, weights_b @ cr_b * 100]
axes[0].bar([0, 1], values, color=[C_MAIN, C_ACC], width=.5)
axes[0].set_xticks([0, 1], ["контроль A", "тест B"])
axes[0].set_title("Агрегат: B выше A")
for i, value in enumerate(values):
    axes[0].text(i, value + .1, f"{value:.1f}%", ha="center")
x = np.arange(2)
axes[1].bar(x-.18, cr_a*100, width=.36, color=C_MAIN, label="контроль A")
axes[1].bar(x+.18, cr_b*100, width=.36, color=C_ACC, label="тест B")
axes[1].set_xticks(x, ["iOS: A 20%, B 80%", "Android: A 80%, B 20%"])
axes[1].set_title("В каждом сегменте: B ниже A на 0,5 п.п.")
axes[1].legend()
axes[0].set_ylabel("конверсия, %")
fig.suptitle("Парадокс Симпсона: сравнение разных смесей не даёт внутрисегментный эффект")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v53_simpson.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 54. Кумулятивные эффекты: сумма релизов vs холдаут
# ============================================================
quarters = np.arange(1, 9)
true_marginals = np.array([3.1, 1.9, 2.4, 1.7, 2.9, 2.0, 2.6, 2.0]) / 100
selected_estimates = np.array([4.2, 2.8, 3.1, 2.4, 3.6, 2.9, 3.3, 2.7]) / 100
declared = (np.cumprod(1 + selected_estimates) - 1) * 100
holdout = (np.cumprod(1 + true_marginals) - 1) * 100
fig, ax = plt.subplots(figsize=(8.4, 4.3))
ax.plot(quarters, declared, "o-", color=C_ACC, label="композиция завышенных оценок (задана)")
ax.plot(quarters, holdout, "o-", color=C_TRUE, label="истинный контраст с исходным продуктом")
ax.fill_between(quarters, holdout, declared, color=C_BAD, alpha=.15)
ax.set_xlabel("релиз"); ax.set_ylabel("прирост против общего baseline, %")
ax.set_title("Учебный сценарий на одной популяции и одном окне:\nистинные последовательные маргиналы перемножаются точно")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v54_holdout_gap.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 55. Winner's curse
# ============================================================
true_eff, n, sd, reps = 2.0, 200, 10.0, 30000
est = rng.normal(true_eff, sd * np.sqrt(2 / n), reps)
significant = est > 1.96 * sd * np.sqrt(2 / n)
fig, ax = plt.subplots(figsize=(8.4, 4.2))
edges = np.histogram_bin_edges(est, bins=80)
ax.hist(est, bins=edges, color="#bbb", alpha=0.8, label=f"все тесты: доля положительных побед {significant.mean():.1%}")
ax.hist(est[significant], bins=edges, color=C_BAD, alpha=0.9, label="отклонённые H0 — «победители»")
ax.axvline(true_eff, color=C_TRUE, lw=3, label=f"истинный эффект = {true_eff}")
ax.axvline(est[significant].mean(), color=C_BAD, lw=2.4, ls="--",
           label=f"средний «победитель» = {est[significant].mean():.1f} (×{est[significant].mean()/true_eff:.1f})")
ax.set_xlabel("оценённый эффект, п.п."); ax.set_ylabel("число симуляций")
ax.set_title("Winner's curse: во что превращается «+2 п.п.» после фильтра значимости")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v55_winners_curse.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 56. Воронка идей: ⅓ выигрывают
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.2))
stages = [("Идеи в тест", 100, C_MAIN), ("Значимый позитив", 33, C_TRUE),
          ("Неопределённые", 50, "#bbb"), ("Значимый негатив", 17, C_BAD)]
xs = np.arange(len(stages))
ax.bar(xs, [s[1] for s in stages], color=[s[2] for s in stages], width=0.6, alpha=0.9)
for i, (name, v, c) in enumerate(stages):
    ax.text(i, v + 2, f"{v}%", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in stages], fontsize=9.5)
ax.set_ylabel("% идей")
ax.set_title("Учебный портфель: исходы тестов; доли не являются универсальными")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v56_ideas_funnel.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 57. Экономика эксперимента
# ============================================================
fig, ax = plt.subplots(figsize=(8.2, 4.2))
mde_pct = np.linspace(0.01, 0.08, 100)
days = 4 * (stats.norm.isf(.025) + stats.norm.ppf(.8)) ** 2 * (2.5 / mde_pct) ** 2 / 100000  # общий N / общий дневной трафик; CV=2.5
value = 40.0 * np.ones_like(mde_pct)  # ценность детекции реального эффекта 4%
cost_delay = 40.0 * 0.05 * days  # 5% ценности теряется за день ожидания (бизнес-оценка)
ax.plot(mde_pct * 100, days, lw=2.6, color=C_MAIN, label="дней теста для этого MDE")
ax.set_xlabel("MDE, % (относительный)")
ax.set_ylabel("дней (при 100к юзеров/день)", color=C_MAIN)
ax2 = ax.twinx()
ax2.plot(mde_pct * 100, cost_delay, lw=2.6, color=C_BAD, label="стоимость задержки роллаута")
ax2.set_ylabel("упущенная ценность, у.е.", color=C_BAD)
ax.set_title("CV=2,5, α=5%, power=80%: меньше MDE — больше дней.\nЗдесь только цена задержки; оптимум требует остальных затрат.")
ax.legend(loc="upper left", fontsize=9); ax2.legend(loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v57_economics.png", bbox_inches="tight"); plt.close(fig)

print("Готово: v46–v57 (12 визуализаций) в", OUTPUT_DIR)
