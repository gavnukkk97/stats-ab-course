from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "course" / "modules" / "M5" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Визуализации для М5 «Байесовские методы»
# Единый стиль: заголовок = вывод. Запуск: python3 scripts/figures/make_visuals_m5.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

rng = np.random.default_rng(505)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_ACC, C_TRUE, C_BAD = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

# ============================================================
# 60. Обновление апостериора данными
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
xs = np.linspace(0, 1, 400)
data_seq = [(0, 0), (6, 2), (34, 16)]
for ax, (s, f) in zip(axes, data_seq):
    a, b = 1 + s, 1 + f
    ax.plot(xs, stats.beta.pdf(xs, 1, 1), color="#bbb", lw=1.6, ls="--", label="априор Beta(1,1)")
    ax.plot(xs, stats.beta.pdf(xs, a, b), color=C_MAIN, lw=2.8,
            label=f"апостериор Beta({a},{b})")
    ax.axvline((s + 1) / (s + f + 2) if s + f else 0.5, color=C_ACC, lw=1.8, ls=":")
    ax.set_title(f"после {s + f} бросков ({s} орлов)", fontsize=10)
    ax.set_xlabel("p (вероятность орла)")
axes[0].set_ylabel("плотность")
axes[0].legend(fontsize=8)
fig.suptitle("Байесовское обучение: вчерашний апостериор = сегодняшний априор, данные заужают уверенность",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v60_posterior_updating.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 61. Априор тает: разные априоры сходятся
# ============================================================
fig, ax = plt.subplots(figsize=(8.4, 4.2))
xs = np.linspace(0, 1, 400)
p_true = 0.7
for (a0, b0), c, name in [((1, 1), "#888", "плоский Beta(1,1)"),
                          ((10, 2), C_ACC, "уверенный «орлячий» Beta(10,2)"),
                          ((2, 10), C_BAD, "уверенный «решечный» Beta(2,10)")]:
    n = 200
    s = rng.binomial(n, p_true)
    ax.plot(xs, stats.beta.pdf(xs, a0, b0), color=c, lw=1.4, ls="--", alpha=0.7)
    ax.plot(xs, stats.beta.pdf(xs, a0 + s, b0 + n - s), color=c, lw=2.6,
            label=f"{name} → после 200 бросков")
ax.axvline(p_true, color="black", lw=2, ls=":")
ax.text(0.705, 12.5, "истина 0.7", fontsize=9, rotation=90)
ax.set_xlabel("p"); ax.set_ylabel("плотность")
ax.set_title("Данные побеждают: противоположные априоры сходятся к одному апостериору")
ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v61_prior_washout.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 62. Base rate fallacy
# ============================================================
fig, ax = plt.subplots(figsize=(8.6, 4.3))
prev = np.linspace(0.0005, 0.5, 400)
sens, spec = 0.95, 0.90
ppv = prev * sens / (prev * sens + (1 - prev) * (1 - spec))
ax.plot(prev * 100, ppv * 100, lw=2.8, color=C_MAIN)
ax.scatter([1], [ppv[100] * 100 if len(prev) > 100 else 8.76], s=90, color=C_BAD, zorder=5)
ax.annotate("редкая болезнь (1%):\nP(болен | +) = 8.8%!", xy=(1, 8.76), xytext=(6, 30),
            arrowprops=dict(arrowstyle="->", color=C_BAD), color=C_BAD, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffe3e3"))
ax.set_xlabel("распространённость, %"); ax.set_ylabel("P(болен | тест+), %")
ax.set_title("Точность теста — не ответ: без базовой частоты «99% точности» ничего не значат")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v62_base_rate.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 63. CI vs credible: интерпретация
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
xs = np.linspace(0, 1, 400)
axes[0].plot(xs, stats.beta.pdf(xs, 58, 62), color=C_MAIN, lw=2.8)
axes[0].fill_between(xs, stats.beta.pdf(xs, 58, 62), where=(xs > 0.4), color=C_MAIN, alpha=0.25)
axes[0].axvline(0.4, color=C_ACC, lw=2.2)
axes[0].set_title("Credible: «с вероятностью 95% p лежит здесь»\n(то, что все хотят услышать)", fontsize=10)
axes[0].set_xlabel("p конверсии")
axes[1].set_xlim(0, 10); axes[1].set_ylim(0, 10); axes[1].axis("off")
axes[1].text(5, 8.5, "Частотный 95% CI", ha="center", fontsize=11, fontweight="bold")
axes[1].text(5, 6.6, "«Если повторить эксперимент\nмного раз, 95% интервалов\nнакроют истинное p»", ha="center", fontsize=10)
axes[1].text(5, 3.6, "…у ЭТОГО интервала вероятности нет:\np либо внутри, либо нет", ha="center", fontsize=9.5,
             color=C_BAD, style="italic")
axes[1].text(5, 1.6, "(мы рисуем ту же картинку,\nно обещаем другое)", ha="center", fontsize=9, color="#666")
fig.suptitle("Два интервала — одна картинка, разные обещания", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v63_ci_vs_credible.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 64. P(B>A) из двух апостериоров
# ============================================================
fig, ax = plt.subplots(figsize=(8.6, 4.3))
xs = np.linspace(.07, .14, 500)
c_A, n_A, c_B, n_B = 1012, 10000, 1063, 10000
a_A, b_A = 1+c_A, 1+n_A-c_A
a_B, b_B = 1+c_B, 1+n_B-c_B
p_pool = (c_A+c_B)/(n_A+n_B)
z = (c_B/n_B-c_A/n_A)/np.sqrt(p_pool*(1-p_pool)*(1/n_A+1/n_B))
p_two = 2*stats.norm.sf(abs(z))
ax.plot(xs, stats.beta.pdf(xs, a_A, b_A), color=C_MAIN, lw=2.8, label=f"A: {c_A}/{n_A} ({c_A/n_A:.2%})")
ax.plot(xs, stats.beta.pdf(xs, a_B, b_B), color=C_ACC, lw=2.8, label=f"B: {c_B}/{n_B} ({c_B/n_B:.2%})")
sA = rng.beta(a_A, b_A, 200000)
sB = rng.beta(a_B, b_B, 200000)
pb = np.mean(sB > sA)
ax.fill_between(xs, stats.beta.pdf(xs, a_B, b_B), where=xs > c_A/n_A, color=C_ACC, alpha=0.15)
ax.set_xlabel("конверсия"); ax.set_ylabel("плотность")
ax.set_title(f"P(B > A) = {pb:.0%} — высказывание, которое понимает продакт\n(двусторонний p-value = {p_two:.3f}; Beta(1,1) prior)")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v64_pb_greater_a.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 65. Expected loss
# ============================================================
diff = sB - sA
loss = np.maximum(-diff, 0)
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.hist(loss * 100, bins=90, color="#bbb", alpha=0.85, label="распределение потерь от роллаута B")
el = loss.mean() * 100
ax.axvline(el, color=C_BAD, lw=3, label=f"Expected Loss = {el:.2f} п.п.")
ax.axvline(0.1, color=C_TRUE, lw=2.4, ls="--", label="threshold of caring = 0.1 п.п.")
verdict = "стоп нельзя" if el > 0.1 else "можно раскатывать"
ax.text(0.97, 0.55, f"EL {'>' if el > 0.1 else '<'} порога → {verdict}",
        transform=ax.transAxes, ha="right", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd"))
ax.set_xlabel("сколько конверсии теряем, если B на самом деле хуже, п.п.")
ax.set_ylabel("число MC-сэмплов")
ax.set_title("Expected loss: решение останавливаться, когда жалко даже ошибиться")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v65_expected_loss.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 66. Thompson sampling: как доля показов эволюционирует
# ============================================================
T, K = 3000, 3
probs = [0.10, 0.12, 0.14]
alpha = np.ones(K); beta_ = np.ones(K)
alloc = np.zeros((T, K))
pulls = np.zeros(K)
for t in range(T):
    theta = rng.beta(alpha, beta_)
    arm = np.argmax(theta)
    r = rng.random() < probs[arm]
    alpha[arm] += r; beta_[arm] += 1 - r
    pulls[arm] += 1
    alloc[t] = pulls
fig, ax = plt.subplots(figsize=(8.6, 4.2))
totals = alloc.sum(axis=1)
for j, (c, lab) in enumerate([("#999", "p=10% (худшая)"), (C_ACC, "p=12%"), (C_TRUE, "p=14% (лучшая)")]):
    ax.plot(alloc[:, j] / np.maximum(totals, 1), color=c, lw=2.2, label=lab)
ax.set_xlabel("показ №"); ax.set_ylabel("доля показов")
ax.set_title("Thompson sampling: трафик сам утекает к лучшей руке — уже к 1000 показов её видно")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v66_thompson_alloc.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 67. Regret: бандит vs A/B
# ============================================================
T, K = 5000, 3
probs = np.array([0.10, 0.12, 0.14])
best = probs.max()
reg_ab, reg_ts, reg_eg = [], [], []
for run in range(300):
    r_ab = r_ts = r_eg = 0.0
    a = np.ones(K); b = np.ones(K); n = np.zeros(K); s = np.zeros(K)
    for t in range(T):
        # A/B: первые 3000 показов равномерно, потом лучший
        arm = t % K if t < 3000 else int(np.argmax(s / np.maximum(n, 1)))
        rew = rng.random() < probs[arm]; r_ab += best - probs[arm]
        n[arm] += 1; s[arm] += rew
        a2 = np.ones(K); b2 = np.ones(K); ns = np.zeros(K); ss = np.zeros(K)
    # ts
    a3 = np.ones(K); b3 = np.ones(K)
    for t in range(T):
        th = rng.beta(a3, b3); arm = int(np.argmax(th))
        rew = rng.random() < probs[arm]; r_ts += best - probs[arm]
        a3[arm] += rew; b3[arm] += 1 - rew
    # eps-greedy
    a4 = np.ones(K); b4 = np.ones(K)
    for t in range(T):
        if rng.random() < 0.1:
            arm = rng.integers(K)
        else:
            arm = int(np.argmax(a4 / (a4 + b4)))
        rew = rng.random() < probs[arm]; r_eg += best - probs[arm]
        a4[arm] += rew; b4[arm] += 1 - rew
    reg_ab.append(r_ab); reg_ts.append(r_ts); reg_eg.append(r_eg)
fig, ax = plt.subplots(figsize=(8.4, 4.2))
ax.bar([0, 1, 2], [np.mean(reg_ab), np.mean(reg_ts), np.mean(reg_eg)],
       color=[C_BAD, C_TRUE, C_ACC], width=0.55, alpha=0.9)
for i, v in enumerate([np.mean(reg_ab), np.mean(reg_ts), np.mean(reg_eg)]):
    ax.text(i, v + 1, f"{v:.0f}", ha="center", fontweight="bold")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Фиксированный A/B\n(3000 показов на тест)", "Thompson\nsampling", "ε-greedy\n(ε=0.1)"])
ax.set_ylabel("regret: потерянные конверсии\nиз 5000 показов (среднее по 300 прогонам)")
ax.set_title("Цена знания: A/B платит за тест, бандит — за скорость учится")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v67_regret.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 68. Partial pooling: shrinkage городов
# ============================================================
n_cities = 14
sizes = np.array([80, 120, 200, 350, 500, 800, 1200, 2000, 3000, 4500, 7000, 10000, 15000, 25000])
p_global = 0.11
p_city = np.clip(rng.normal(p_global, 0.025, n_cities), 0.02, 0.25)
conv = rng.binomial(sizes, p_city)
raw = conv / sizes
k = 400  # «сила» общей подушки
shrunk = (conv + k * p_global) / (sizes + k)
fig, ax = plt.subplots(figsize=(8.6, 4.5))
order = np.argsort(raw)
for i in order:
    c = plt.cm.viridis(np.log10(sizes[i]) / np.log10(25000))
    ax.plot([raw[i], shrunk[i]], [i, i], color="#ccc", lw=1.2, zorder=1)
ax.scatter(raw[order], np.arange(n_cities), s=30 + sizes / 80, c=[plt.cm.viridis(np.log10(s) / np.log10(25000)) for s in sizes[order]],
           zorder=3, label="сырая конверсия (размер = n города)")
ax.scatter(shrunk[order], np.arange(n_cities), s=30 + sizes / 80, facecolors="none",
           edgecolors=C_BAD, linewidths=2, zorder=3, label="после shrinkage")
ax.axvline(p_global, color="black", ls="--", lw=1.8)
ax.text(p_global + 0.001, 0.5, "глобальное\nсреднее", fontsize=9)
ax.set_xlabel("конверсия города"); ax.set_ylabel("города (отсортированы по сырой)")
ax.set_title("Partial pooling: маленькие города тянутся к среднему, большие остаются собой")
ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v68_shrinkage.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 69. MCMC: trace plot и сходимость (концепт)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
chain1 = np.concatenate([rng.normal(0, 1, 200) * np.linspace(3, 1, 200), rng.normal(0.5, 1, 1800)])
chain2 = np.concatenate([rng.normal(1.2, 1, 200) * np.linspace(2, 1, 200), rng.normal(0.5, 1, 1800)])
axes[0].plot(chain1, lw=0.7, color=C_MAIN, alpha=0.85)
axes[0].plot(chain2, lw=0.7, color=C_ACC, alpha=0.85)
axes[0].axvline(200, color="gray", ls=":", lw=1.4)
axes[0].text(220, 3.6, "burn-in выбрасываем", fontsize=8.5)
axes[0].set_title("Trace plot: две цепи стартовали из разных мест — сошлись\n(«гусеница» без тренда = здоровье)", fontsize=10)
axes[0].set_xlabel("итерация"); axes[0].set_ylabel("значение параметра")
axes[1].hist(chain1[200:], bins=60, color=C_MAIN, alpha=0.6, density=True, label="цепь 1")
axes[1].hist(chain2[200:], bins=60, color=C_ACC, alpha=0.6, density=True, label="цепь 2")
axes[1].set_title("Апостериор из MCMC: распределение собрали сэмплами", fontsize=10)
axes[1].legend(fontsize=9)
fig.suptitle("MCMC на пальцах: блуждание по параметрам, из которого складывается апостериор",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v69_mcmc_trace.png", bbox_inches="tight"); plt.close(fig)

print("Готово: v60–v69 (10 визуализаций) в", OUTPUT_DIR)
