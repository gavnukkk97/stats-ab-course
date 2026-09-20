from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "course" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Карта маршрута по курсу: генератор + автопроверка layout.
# Запуск: python3 scripts/figures/make_visuals_nav.py  → course_route.png + "LAYOUT OK"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(15.5, 7.2))
ax.set_xlim(0, 30); ax.set_ylim(0, 14); ax.axis("off")

placed = []  # (имя, rect, text) для автопроверки

def box(name, x, y, w, h, lines, fc, fs=9.5):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#555", lw=1.8))
    t = ax.text(x + w/2, y + h/2, lines, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="#222", linespacing=1.6)
    placed.append((name, (x, y, w, h), t))

def elbow(points, color="#222", lw=2.2, ls="-"):
    """Ортогональная ломаная со стрелкой на конце."""
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round")
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color))

C_CORE, C_BAYES, C_CAUSAL, C_PLAT, C_START = "#E8F0FE", "#F3E8FD", "#FFF3CD", "#E5F0F7", "#F2F2F2"

# --- основной коридор М1→М4 ---
Y0, H, W = 6.2, 2.0, 5.0
box("m1", 0.6, Y0, W, H, "М1 · Базовая статистика\n4 урока · ~2 нед\nзадания M1", C_CORE, fs=9.2)
box("m2", 6.15, Y0, W, H, "М2 · Проверка гипотез\n5 уроков · ~3 нед\nкейсы: 12·18", C_CORE, fs=9.2)
box("m3", 11.7, Y0, W, H, "М3 · Статистика для АБ\n7 уроков · ~3 нед\nкейсы: 01·03·09·11·16", C_CORE, fs=9.2)
box("m4", 17.25, Y0, W, H, "М4 · АБ на практике\n6 уроков · ~3 нед\nкейсы: см. навигацию", C_CORE, fs=9.2)

elbow([(5.6, 7.2), (6.15, 7.2)])
elbow([(11.15, 7.2), (11.7, 7.2)])
elbow([(16.7, 7.2), (17.25, 7.2)])

# --- блоки по интересам ---
box("m5", 23.4, 10.2, 5.2, 2.0, "М5 · Байесовские методы\n5 уроков · ~2 нед\nкейсы: 05", C_BAYES, fs=9.2)
box("m6", 23.4, 6.2, 5.2, 2.0, "М6 · Каузальный вывод\n5 уроков · ~2 нед\nкейсы: 02·17", C_CAUSAL, fs=9.2)
box("m7", 17.25, 1.4, 5.2, 2.0, "М7 · АБ-платформа\n4 урока · ~1–2 нед\nкейсы: 08", C_PLAT, fs=9.2)

elbow([(14.2, 8.2), (14.2, 11.2), (23.4, 11.2)])   # М3 → М5
elbow([(22.25, 7.2), (23.4, 7.2)])                 # М4 → М6
elbow([(19.75, 6.2), (19.75, 3.4)])                # М4 → М7

# --- старт и альтернативные треки ---
box("start", 0.6, 1.4, 4.4, 2.0, "СТАРТ\nцикл + самотест\n(course/NAVIGATION.md)", C_START, fs=9)

elbow([(2.8, 3.4), (2.8, 4.6), (16.95, 4.6), (16.95, 7.2), (17.25, 7.2)],
      color="#DD8452", lw=2.0, ls="--")
ax.text(9.8, 4.85, "трек продакта: без формул — 1.1, 1.3, 2.1, 3.1 → М4 + кейсы",
        fontsize=9, color="#B5651D", ha="center", fontweight="bold")

arc = FancyArrowPatch((5.0, 2.4), (14.2, 6.2), connectionstyle="arc3,rad=-0.25",
                      arrowstyle="-|>", mutation_scale=18, lw=2.0, ls="--", color="#55A868")
ax.add_patch(arc)
ax.text(5.6, 0.9, "экспресс: знаешь матстатистику — сразу в М3 (3.1–3.3, 3.5, 3.7)",
        fontsize=9, color="#2d6a4f", ha="left", fontweight="bold")

ax.text(15.0, 13.4, "Карта маршрута: основной коридор М1→М4, затем блоки по интересам",
        ha="center", fontsize=14, fontweight="bold")
ax.text(15.0, 12.7, "ядро → самостоятельный проект; сроки зависят от подготовки; специализации отдельно",
        ha="center", fontsize=10, color="#555", style="italic")

ax.text(24.0, 3.2, "— опорные пункты\n-- зелёный: экспресс\n-- оранжевый: продакт", fontsize=9, color="#555", va="top")

elbow([(2.8, 3.4), (2.8, 6.2)])
fig.savefig(OUTPUT_DIR / "course_route.png", bbox_inches="tight", dpi=150)

# --- автопроверка: тексты внутри боксов, боксы не пересекаются ---
fig.canvas.draw()
inv = ax.transData.inverted()
problems, texts = [], []
for name, (x, y, w, h), t in placed:
    bb = t.get_window_extent(fig.canvas.get_renderer())
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    texts.append((name, x0, y0, x1, y1))
    if x0 < x - 0.05 or x1 > x + w + 0.05:
        problems.append(f"[{name}] текст шире бокса: [{x0:.2f}, {x1:.2f}] vs [{x:.2f}, {x+w:.2f}]")
    if y0 < y - 0.05 or y1 > y + h + 0.05:
        problems.append(f"[{name}] текст выше бокса: [{y0:.2f}, {y1:.2f}] vs [{y:.2f}, {y+h:.2f}]")
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        n1, a0, b0, a1, b1 = texts[i]; n2, c0, d0, c1, d1 = texts[j]
        if a0 < c1 and c0 < a1 and b0 < d1 and d0 < b1:
            problems.append(f"тексты пересекаются: {n1} <-> {n2}")
for i in range(len(placed)):
    for j in range(i + 1, len(placed)):
        n1, (x, y, w, h), _ = placed[i]; n2, (x2, y2, w2, h2), _ = placed[j]
        if x < x2 + w2 and x2 < x + w and y < y2 + h2 and y2 < y + h:
            problems.append(f"боксы пересекаются: {n1} <-> {n2}")
if problems:
    raise SystemExit("LAYOUT FAIL:\n" + "\n".join(problems))
print("LAYOUT OK")
