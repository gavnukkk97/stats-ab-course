# %% [markdown]
# # Визуализации для М4 «АБ-тесты на практике»
# Единый стиль: заголовок = вывод. Запуск: python3 make_visuals_m4.py
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
    (0.7, 4.5, 8.6, C_ACC, "Guardrails (ограничения)", "не должны упасть: выручка, латентность, отписки, поддержка"),
    (0.2, 2.5, 9.6, "#8C8C8C", "Инварианты (здоровье)", "не должны меняться: скорость логов, SRM, размер групп"),
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
fig.savefig("v46_oec_pyramid.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 47. Decision rule 2×2
# ============================================================
fig, ax = plt.subplots(figsize=(8.4, 4.9))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
cells = [
    (0.3, 3.1, "#9CC5E8", "Зелёный: эффект > 0, значимо", "ROLL OUT", "+ CI эффекта в отчёт"),
    (5.3, 3.1, "#F2B8B5", "Красный: эффект < 0, значимо", "ОТКАТ + разбирательство", "guardrail или гипотеза не та"),
    (0.3, 0.3, "#EDE7DC", "Серый: эффект > 0, не значимо", "нет решения", "продлить / поднять мощность / iterate"),
    (5.3, 0.3, "#EDE7DC", "Серый: эффект < 0, не значимо", "нет решения", "скорее нет эффекта — но не доказано"),
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
ax.set_title("Decision rule 2×2: решение принимается по знаку × значимости OEC (заранее, в дизайн-доке)")
fig.savefig("v47_decision_rule.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 48. Сплит-система: слои и слоты
# ============================================================
fig, ax = plt.subplots(figsize=(9, 4.7))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.text(5, 5.7, "Слои = независимые лотереи: одна соль на слой", ha="center", fontsize=11.5, fontweight="bold")
layers = [
    (4.0, "слой 1: поиск/рекомендации", "соль S1", ["тест А", "контроль"]),
    (2.3, "слой 2: доставка", "соль S2", ["тест B", "тест C", "контроль"]),
    (0.6, "слой 3: промо", "соль S3", ["тест D", "контроль"]),
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
ax.text(5, 0.1, "юзер участвует в каждом слое независимо → десятки параллельных тестов без пересечения трафика",
        ha="center", fontsize=9, style="italic", color="#444",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
fig.savefig("v48_split_layers.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 49. SRM: какие отклонения ловит хи-квадрат
# ============================================================
cases = [(49500, 50500), (49995, 50005), (49800, 50200), (495000, 505000)]
fig, ax = plt.subplots(figsize=(9, 4.2))
labels, pvals = [], []
for a, b in cases:
    n = a + b
    chi2 = (a - n / 2) ** 2 / (n / 2) + (b - n / 2) ** 2 / (n / 2)
    p = 1 - stats.chi2.cdf(chi2, df=1)
    labels.append(f"{a//1000}к/{b//1000}к" if a >= 100000 else f"{a}/{b}")
    pvals.append(max(p, 1e-30))
colors = [C_BAD if p < 0.001 else C_TRUE for p in pvals]
ax.bar(range(len(cases)), [max(p, 1e-25) for p in pvals], color=colors, width=0.55, alpha=0.9)
ax.set_yscale("log"); ax.axhline(0.001, color="black", ls="--", lw=1.8)
ax.text(len(cases) - 0.4, 0.0016, "порог SRM: p < 0.001", fontsize=9.5, ha="right")
ax.set_xticks(range(len(cases))); ax.set_xticklabels(labels)
for i, p in enumerate(pvals):
    verdict = "SRM!" if p < 0.001 else "норма"
    ax.text(i, max(p, 1e-25) * 1.3, f"p={p:.1e}\n{verdict}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("p-value хи-квадрат (log)")
ax.set_title("SRM-детектор: на миллионных выборках ловится перекос в 0,5% — это не шум, тест невалиден")
fig.tight_layout(); fig.savefig("v49_srm.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 50. A/A-тесты: здоровье платформы
# ============================================================
pvals = rng.uniform(size=2000)
pvals_srm = np.concatenate([rng.uniform(size=1860), rng.uniform(0, 0.02, 140)])
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
axes[0].hist(pvals, bins=40, color=C_TRUE, alpha=0.85)
axes[0].set_title("Здоровая сплит-система: 2000 A/A-тестов\np-value равномерен (доля <0.05 ≈ 5%)")
axes[0].set_xlabel("p-value")
axes[1].hist(pvals_srm, bins=40, color=C_BAD, alpha=0.85)
axes[1].set_title("Сломанная (фильтр новинок): p скапливаются у нуля\n7% ложных «эффектов» — платформа врёт")
axes[1].set_xlabel("p-value")
fig.suptitle("A/A-тест — ЭКГ платформы экспериментов", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("v50_aa_health.png", bbox_inches="tight"); plt.close(fig)

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
axes[0].text(5, 0.35, "весь город — одна группа: курьеры не смешиваются", ha="center", fontsize=9, style="italic")
t = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб"]
for i, d in enumerate(t):
    x = 0.7 + i * 1.55
    c = cols[i % 2]
    axes[1].add_patch(plt.Rectangle((x, 2.0), 1.35, 2.2, fc=c, alpha=0.35, ec=c, lw=2))
    axes[1].text(x + 0.67, 4.4, d, ha="center", fontsize=9)
    axes[1].text(x + 0.67, 3.1, "T" if i % 2 == 0 else "C", ha="center", fontsize=12, fontweight="bold")
axes[1].text(5, 1.2, "вся платформа переключается: T-C-T-C…\nспилловер умирает на границе дня", ha="center", fontsize=9, style="italic")
fig.suptitle("Двусторонний рынок: обычный АБ течёт — лечат кластерные дизайны", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("v51_interference_designs.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 52. Эффект новизны
# ============================================================
weeks = np.arange(1, 9)
novelty = 6.5 * np.exp(-0.45 * weeks) + 1.0
fig, ax = plt.subplots(figsize=(8, 4.1))
ax.plot(weeks, novelty, "o-", lw=2.6, color=C_MAIN, ms=8, label="эффект по неделям")
ax.axhline(1.0, color=C_TRUE, ls="--", lw=2, label="долгосрочный эффект")
ax.axvspan(1, 2, color=C_BAD, alpha=0.12)
ax.text(1.06, 5.6, "только 2 недели:\nтест увидит ×3 долгосрочного", fontsize=9.5, color=C_BAD, fontweight="bold")
ax.set_xlabel("неделя после раскатки"); ax.set_ylabel("эффект, %")
ax.set_title("Эффект новизны: вздёрнутый старт затухает — короткий тест переоценит фичу")
ax.legend()
fig.tight_layout(); fig.savefig("v52_novelty.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 53. Парадокс Симпсона
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
seg = [("iOS", 8000, 0.12, 0.115, C_MAIN), ("Android", 2000, 0.08, 0.075, C_ACC)]
for ax, mode in [(axes[0], "Агрегировано"), (axes[1], "По сегментам (платформа)")]:
    ax.set_title(mode, fontsize=11)
    ax.set_xlabel("конверсия")
    ax.axhline(0, color="gray", lw=0.5)
axb = axes[0]
tot_a = sum(n * ca for _, n, ca, _, _ in seg) / sum(n for _, n, _, _, _ in seg)
tot_b = sum(n * cb for _, n, _, cb, _ in seg) / sum(n for _, n, _, _, _ in seg)
axb.bar([0, 1], [tot_a * 100, tot_b * 100], color=[C_TRUE, C_BAD], width=0.5)
for i, v in enumerate([tot_a * 100, tot_b * 100]):
    axb.text(i, v + 0.05, f"{v:.1f}%", ha="center", fontweight="bold")
axb.set_xticks([0, 1]); axb.set_xticklabels(["тест", "контроль"])
axb.text(0.5, -0.9, "тест выигрывает!", ha="center", fontsize=11, color=C_TRUE, fontweight="bold")
w = 0.35
for i, (name, n, ca, cb, c) in enumerate(seg):
    axes[1].bar([i - w / 2, i + w / 2], [ca * 100, cb * 100], color=[C_TRUE, C_BAD], width=w)
    axes[1].text(i - w / 2, ca * 100 + 0.05, f"{ca*100:.1f}", ha="center", fontsize=9)
    axes[1].text(i + w / 2, cb * 100 + 0.05, f"{cb*100:.1f}", ha="center", fontsize=9)
axes[1].set_xticks(range(2)); axes[1].set_xticklabels(["iOS (80%)", "Android (20%)"])
axes[1].text(0.5, -0.9, "в обоих сегментах тест ПРОИГРЫВАЕТ\n(в тесте просто больше iOS)",
             ha="center", fontsize=10, color=C_BAD, fontweight="bold")
fig.suptitle("Парадокс Симпсона: агрегат врёт, когда микс сегментов различается", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("v53_simpson.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 54. Кумулятивные эффекты: сумма релизов vs холдаут
# ============================================================
quarters = np.arange(1, 9)
declared = np.cumsum(np.array([4.2, 2.8, 3.1, 2.4, 3.6, 2.9, 3.3, 2.7]))
holdout = np.cumsum(np.array([3.1, 1.9, 2.4, 1.7, 2.9, 2.0, 2.6, 2.0]))
fig, ax = plt.subplots(figsize=(8.4, 4.3))
ax.plot(quarters, declared, "o-", lw=2.6, color=C_ACC, label="сумма эффектов релизов (по АБ-тестам)")
ax.plot(quarters, holdout, "o-", lw=2.6, color=C_TRUE, label="прирост vs глобальный холдаут")
ax.fill_between(quarters, holdout, declared, color=C_BAD, alpha=0.15)
ax.text(4.5, declared[3] - 1.5, "разрыв = winner's curse\n+ взаимодействия фич",
        fontsize=10, color=C_BAD, fontweight="bold")
ax.set_xlabel("квартал"); ax.set_ylabel("кумулятивный прирост, %")
ax.set_title("Кейс Airbnb: сумма релизов +40,7%, по холдауту +30,6% — треть «прироста» не существует")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig("v54_holdout_gap.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 55. Winner's curse
# ============================================================
true_eff, n, sd, reps = 2.0, 4000, 10.0, 30000
est = rng.normal(true_eff, sd * np.sqrt(2 / n), reps)
significant = est > 1.96 * sd * np.sqrt(2 / n)
fig, ax = plt.subplots(figsize=(8.4, 4.2))
ax.hist(est, bins=80, color="#bbb", alpha=0.8, label="все тесты (эффект есть, но мощность ~50%)")
ax.hist(est[significant], bins=40, color=C_BAD, alpha=0.9, label="отклонённые H0 — «победители»")
ax.axvline(true_eff, color=C_TRUE, lw=3, label=f"истинный эффект = {true_eff}")
ax.axvline(est[significant].mean(), color=C_BAD, lw=2.4, ls="--",
           label=f"средний «победитель» = {est[significant].mean():.1f} (×{est[significant].mean()/true_eff:.1f})")
ax.set_xlabel("оценённый эффект"); ax.set_ylabel("плотность")
ax.set_title("Winner's curse: во что превращается «+2%» после фильтра значимости")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig("v55_winners_curse.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 56. Воронка идей: ⅓ выигрывают
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.2))
stages = [("Идеи в тест", 100, C_MAIN), ("Значимый позитив", 33, C_TRUE),
          ("Серые / нет эффекта", 50, "#bbb"), ("Значимый негатив", 17, C_BAD)]
xs = np.arange(len(stages))
ax.bar(xs, [s[1] for s in stages], color=[s[2] for s in stages], width=0.6, alpha=0.9)
for i, (name, v, c) in enumerate(stages):
    ax.text(i, v + 2, f"{v}%", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in stages], fontsize=9.5)
ax.set_ylabel("% идей")
ax.set_title("«⅓ идей выигрывают» (Microsoft, KDD'13): эксперименты — машина по ОТКЛОНЕНИЮ идей")
fig.tight_layout(); fig.savefig("v56_ideas_funnel.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 57. Экономика эксперимента
# ============================================================
fig, ax = plt.subplots(figsize=(8.2, 4.2))
mde_pct = np.linspace(0.05, 0.5, 100)
days = (1.96 + 0.84) ** 2 / (mde_pct * np.sqrt(2)) ** 2 * 1000 / 100000  # n/(трафик в день) в днях
value = 40.0 * np.ones_like(mde_pct)  # ценность детекции реального эффекта 4%
cost_delay = 40.0 * 0.05 * days  # 5% ценности теряется за день ожидания (бизнес-оценка)
ax.plot(mde_pct * 100, days, lw=2.6, color=C_MAIN, label="дней теста для этого MDE")
ax.set_xlabel("MDE, % (относительный)")
ax.set_ylabel("дней (при 100к юзеров/день)", color=C_MAIN)
ax2 = ax.twinx()
ax2.plot(mde_pct * 100, cost_delay, lw=2.6, color=C_BAD, label="стоимость задержки роллаута")
ax2.set_ylabel("упущенная ценность, у.е.", color=C_BAD)
ax.set_title("Экономика: чем меньше MDE — тем дольше тест; где-то есть оптимум")
ax.legend(loc="upper left", fontsize=9); ax2.legend(loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig("v57_economics.png", bbox_inches="tight"); plt.close(fig)

print("Готово: v46–v57 (12 визуализаций) в", __file__.rsplit("/", 1)[0])
