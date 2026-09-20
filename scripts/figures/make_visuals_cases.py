from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "course" / "cases" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Иллюстрация бинарной модели кейса 12; это НЕ оценка win rate компаний.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.2, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})
alpha, power = .05, .8
priors = np.array([.1, .2, .5])
true_rejections = priors * power
false_rejections = (1 - priors) * alpha
non_rejections = 1 - true_rejections - false_rejections
fig, ax = plt.subplots(figsize=(9.2, 4.6))
y = np.arange(len(priors))
left = np.zeros(len(priors))
for rates, label, color in [
    (true_rejections, "Отвержение при H₁", "#55A868"),
    (false_rejections, "Отвержение при H₀ (ошибка)", "#C44E52"),
    (non_rejections, "H₀ не отвергнута", "#8C8C8C"),
]:
    ax.barh(y, 100 * rates, left=100 * left, color=color, ec="white", height=.55, label=label)
    for row, (start, rate) in enumerate(zip(left, rates)):
        ax.text(100 * (start + rate / 2), row, f"{100 * rate:g}%", ha="center", va="center",
                color="white", fontsize=7.5 if rate < .04 else 9, fontweight="bold")
    left += rates
ax.set_yticks(y)
ax.set_yticklabels([f"π = {p:.0%} истинных H₁" for p in priors])
ax.set_xlim(0, 100)
ax.set_xlabel("Ожидаемая доля всех проверок, %")
ax.set_title("Учебная модель: доля отвержений не равна доле реальных эффектов\nМощность = 80%, α = 5%; H₀ не отвергнута ≠ эффект отсутствует")
ax.invert_yaxis()
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.20), ncol=1, frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cases_win_rate.png", bbox_inches="tight")
plt.close(fig)
print("saved cases_win_rate.png: synthetic binary model, not company statistics")
