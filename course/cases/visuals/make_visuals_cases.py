# %% [markdown]
# # Визуализации для библиотеки кейсов
# Единый стиль: заголовок = вывод. Запуск: python3 make_visuals_cases.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_ACC, C_TRUE, C_BAD = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

# ============================================================
# К1. Судьба идей: сколько гипотез реально работают
# ============================================================
fig, ax = plt.subplots(figsize=(9.2, 4.4))
rows = [
    ("Microsoft / Bing\n(Kohavi, KDD'13)", 33, 33, 34),
    ("Google\n(ранние годы)", 15, 55, 30),
    ("Авито / Trisigma\n(интервью 2026)", 10, 80, 10),
]
labels = ["улучшили метрики", "без эффекта", "ухудшили"]
colors = [C_TRUE, "#8C8C8C", C_BAD]
y = np.arange(len(rows))
left = np.zeros(len(rows))
for j, (lab, col) in enumerate(zip(labels, colors)):
    vals = np.array([r[1 + j] for r in rows], dtype=float)
    ax.barh(y, vals, left=left, color=col, alpha=0.85, ec="white", lw=1.5,
            label=lab, height=0.62)
    for i, v in enumerate(vals):
        if v >= 8:
            ax.text(left[i] + v / 2, y[i], f"{v:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=10)
    left += vals
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=10)
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("доля проверенных идей, %")
ax.invert_yaxis()
ax.set_title("Большинство идей не работают — поэтому платформа экспериментов экономит деньги")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
fig.tight_layout()
fig.savefig("cases_win_rate.png", bbox_inches="tight")
print("saved cases_win_rate.png")
