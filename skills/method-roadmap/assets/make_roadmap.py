# Дорожная карта выбора метода: генератор + автопроверка layout.
# Запуск: python3 make_roadmap.py  → decision_roadmap.png + "LAYOUT OK"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q_C, M_C, AB_C, BAD_C = "#FFF3CD", "#E8F0FE", "#D8F3DC", "#FFE3E3"
BAD_T, DK, GR = "#C44E52", "#222222", "#555555"

fig, ax = plt.subplots(figsize=(13.5, 9.4))
ax.set_xlim(0, 27); ax.set_ylim(0, 19); ax.axis("off")

placed = []  # (имя, rect, text-artist) для автопроверки

def box(name, x, y, w, h, text, fc, fs=11, bold=True, ec="#555", tc=DK):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.8))
    t = ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=tc, linespacing=1.45)
    placed.append((name, (x, y, w, h), t))

def harrow(x1, x2, y, color=DK, label=None):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color=color))
    if label:
        ax.text((x1 + x2)/2, y + 0.3, label, ha="center", fontsize=11,
                fontweight="bold", color=color)

def varrow(x, y1, y2, color=DK, label=None, fs=10.5):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color=color))
    if label:
        ax.text(x + 0.3, (y1 + y2)/2, label, ha="left", va="center",
                fontsize=fs, fontweight="bold", color=color)

ax.text(13.5, 18.4, "Как на практике выбирают метод оценки эффекта",
        ha="center", fontsize=17, fontweight="bold")
ax.text(13.5, 17.6, "дорожная карта курса «Проверка гипотез» · спускайся сверху вниз, не перескакивая",
        ha="center", fontsize=11.5, color=GR, style="italic")

QX, QW = 1.0, 8.6
MX, MW = 13.5, 12.0
MID_Q = QX + QW / 2

# Строка 1: рандомизация доступна → A/B
box("q1", QX, 14.3, QW, 2.2,
    "1. Можно рандомизировать?\n(часть юзеров видит изменение,\nчасть — нет)", Q_C, fs=11.5)
harrow(9.6, MX, 15.4, "#2d6a4f", label="ДА")
box("ab", MX, 13.9, MW, 3.0,
    "A/B-ТЕСТ — 90% кейсов на практике\nюниты · слои · SRM · guardrails\nсеть → geo/switchback · ждать дорого → sequential",
    AB_C, fs=11, ec="#2d6a4f")
varrow(MID_Q, 14.3, 12.7, BAD_T,
       label="НЕТ: этика / дорого /\nуже произошло /\nэффект на всех сразу")

# Строка 2: есть контроль и пре-период → DiD
box("q2", QX, 10.4, QW, 2.3,
    "2. Есть нетронутая группа\nи данные ДО вмешательства?", Q_C, fs=11.5)
harrow(9.6, MX, 11.55, DK)
box("did", MX, 10.7, MW, 1.7,
    "DiD — разность разностей\n(+ event study: проверь параллельные тренды)", M_C, fs=11)
box("did+", MX, 8.7, MW, 1.6,
    "группы неоднородны → PSM+DiD / DR-DiD\nstaggered → Callaway & Sant'Anna (TWFE врёт)",
    M_C, fs=9.5, bold=False)
varrow(MX + MW/2, 10.7, 10.3, GR)
varrow(MID_Q, 10.4, 8.0, BAD_T, label="НЕТ")

# Строка 3: пороговое правило → RDD
box("q3", QX, 5.7, QW, 2.3,
    "3. Решение принимается\nпо ПОРОГУ?\n(балл, возраст, рейтинг)", Q_C, fs=11.5)
harrow(9.6, MX, 6.85, DK)
box("rdd", MX, 5.7, MW, 2.3,
    "RDD — сравниваем 74 и 76 баллов:\n«почти близнецы» по обе стороны порога\nпроверки: плотность у порога, плацебо-пороги",
    M_C, fs=11)
varrow(MID_Q, 5.7, 4.2, BAD_T, label="НЕТ")

# Строка 4: остаточные ситуации — «гребёнка» на три одинаковых бокса
box("q4", QX, 1.7, QW, 2.0,
    "4. Что осталось?\n(один лечёный юнит · только\nковариаты · внешний инструмент)", Q_C, fs=11)
ROW4_Y, ROW4_H = 1.7, 2.0
boxes4 = [
    ("sc",   13.50, 4.30, "Synthetic control\n+ ITS · плацебо-юниты", 9.0),
    ("psm",  18.05, 4.30, "PSM / IPW / DR\nбаланс SMD < 0.1", 9.8),
    ("iv",   22.60, 4.30, "IV (Wald / 2SLS)\nинструмент Z, F > 10", 9.8),
]
CONN_Y = 4.35
ax.plot([9.6, 24.75], [CONN_Y, CONN_Y], color=DK, lw=2.2)
ax.plot([9.6, 9.6], [2.7, CONN_Y], color=DK, lw=2.2)
for name, bx, bw, txt, fs4 in boxes4:
    varrow(bx + bw/2, CONN_Y, ROW4_Y + ROW4_H, DK)
    box(name, bx, ROW4_Y, bw, ROW4_H, txt, M_C, fs=fs4)

# Финальный баннер
box("banner", QX, 0.12, 24.5, 1.30,
    "Ничего не подошло → НЕ ДЕЛАЙ выводов: собирай данные\nи жди возможности эксперимента. «Оценка при допущениях — не доказательство»",
    BAD_C, fs=11, ec=BAD_T, tc=BAD_T)
varrow(MID_Q, 1.7, 1.42, BAD_T)

fig.savefig("decision_roadmap.png", bbox_inches="tight", dpi=150)

# --- автопроверка: каждый текст внутри своего бокса, ничего не пересекается ---
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
