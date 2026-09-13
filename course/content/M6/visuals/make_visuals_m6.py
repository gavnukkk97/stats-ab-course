# %% [markdown]
# # Визуализации для М6 «Каузальный вывод»
# Единый стиль: заголовок = вывод. Запуск: python3 make_visuals_m6.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

rng = np.random.default_rng(606)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_ACC, C_TRUE, C_BAD = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

def node(ax, x, y, label, c=C_MAIN):
    ax.scatter([x], [y], s=2200, color=c, alpha=0.25, zorder=2)
    ax.scatter([x], [y], s=2200, facecolors="none", edgecolors=c, lw=2.2, zorder=3)
    ax.text(x, y, label, ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)

def edge(ax, x1, y1, x2, y2, c="#444", lw=2.4, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=lw, color=c, linestyle=ls))

# ============================================================
# 72. Три структуры: fork / chain / collider
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
# fork
node(axes[0], 5, 8, "Z", C_ACC)
node(axes[0], 2.5, 3, "X"); node(axes[0], 7.5, 3, "Y")
edge(axes[0], 4.6, 7.3, 3.0, 3.9, C_ACC); edge(axes[0], 5.4, 7.3, 7.0, 3.9, C_ACC)
axes[0].set_title("Fork (конфаундер)\nZ — общая причина X и Y\nконтролировать: ДА", fontsize=10)
# chain
node(axes[1], 2.5, 8, "X"); node(axes[1], 5, 4.5, "M", C_ACC); node(axes[1], 7.5, 1.5, "Y")
edge(axes[1], 3.1, 7.4, 4.4, 5.3); edge(axes[1], 5.6, 3.9, 6.9, 2.3)
axes[1].set_title("Chain (медиатор)\nX → M → Y\nконтролировать = «убить» часть эффекта", fontsize=10)
# collider
node(axes[2], 2.5, 8, "X"); node(axes[2], 7.5, 8, "Y")
node(axes[2], 5, 3, "S", C_ACC)
edge(axes[2], 3.2, 7.3, 4.4, 4.0, C_ACC); edge(axes[2], 6.8, 7.3, 5.6, 4.0, C_ACC)
axes[2].text(5, 5.6, "контролировать S\n= рождает ложную связь X–Y", ha="center",
             fontsize=9, color=C_BAD, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="#ffe3e3"))
axes[2].set_title("Collider (коллайдер)\nS — общее следствие", fontsize=10)
fig.suptitle("Три структуры DAG — три разных решения «контролировать или нет»", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("v72_dag_structures.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 73. Симпсон с DAG
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
seg = [("новый UI", 6000, 0.061, 0.065), ("старый UI", 6000, 0.102, 0.098)]
for i, (name, n, ca, cb) in enumerate(seg):
    axes[0].bar([i - 0.18, i + 0.18], [ca * 100, cb * 100],
                color=[C_TRUE, C_BAD], width=0.32)
    axes[0].text(i - 0.18, ca * 100 + 0.12, f"{ca*100:.1f}", ha="center", fontsize=9)
    axes[0].text(i + 0.18, cb * 100 + 0.12, f"{cb*100:.1f}", ha="center", fontsize=9)
axes[0].set_xticks(range(2)); axes[0].set_xticklabels(["новая платформа\n(богатые юзеры)", "старая платформа\n(бюджетные юзеры)"])
axes[0].set_ylabel("конверсия, %")
axes[0].set_title("Внутри каждой платформы тест ХУЖЕ\n(зелёный — контроль)", fontsize=10)
axes[0].legend([mpatches.Patch(color=C_TRUE), mpatches.Patch(color=C_BAD)],
               ["контроль", "тест"], fontsize=8, loc="upper left")
agg_a = sum(n * ca for _, n, ca, _ in seg) / sum(n for _, n, _, _ in seg)
agg_b = sum(n * cb for _, n, _, cb in seg) / sum(n for _, n, _, _ in seg)
axes[0].text(0.5, -0.28, f"агрегат: тест {agg_b*100:.1f}% > контроль {agg_a*100:.1f}% — «тест выигрывает»",
             transform=axes[0].transAxes, ha="center", fontsize=10, color=C_BAD, fontweight="bold")
ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
node(ax, 5, 8.5, "P", C_ACC); node(ax, 2.5, 3.5, "D"); node(ax, 7.5, 3.5, "Y")
edge(ax, 4.5, 7.8, 3.1, 4.3, C_ACC); edge(ax, 5.5, 7.8, 6.9, 4.3, C_ACC)
edge(ax, 3.3, 3.5, 6.7, 3.5)
ax.text(5, 6.3, "P = платформа\n(конфаундер: и дизайн,\nи конверсия зависят от неё)",
        ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
ax.text(5, 1.5, "D = новый дизайн, Y = конверсия", ha="center", fontsize=9, color="#555")
fig.suptitle("Парадокс Симпсона: агрегат «выигрывает», потому что тест попал на богатую платформу",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig("v73_simpson_dag.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 74. Potential outcomes: два мира
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 4.2))
n = 24
y0 = rng.normal(100, 12, n)
y1 = y0 + rng.normal(28, 8, n)
for i in range(n):
    ax.plot([0, 1], [y0[i], y1[i]], color="#ccc", lw=0.8, zorder=1)
obs_t = rng.random(n) < 0.5
ax.scatter(np.zeros(n) - 0.13 + 0.26 * obs_t, np.where(obs_t, y1, y0), s=70, color=C_BAD, zorder=3)
ax.scatter(np.zeros(n) - 0.13 + 0.26 * (~obs_t), np.where(obs_t, y1, y0), s=70, color=C_MAIN, alpha=0.35, zorder=2)
ax.text(0, 145, "мир БЕЗ воздействия\nY(0)", ha="center", fontsize=11, fontweight="bold")
ax.text(1, 145, "мир С воздействием\nY(1)", ha="center", fontsize=11, fontweight="bold")
ax.text(0.5, 62, "эффект каждого = разность его миров\n(наблюдаем только один — закрашенный)",
        ha="center", fontsize=10, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd"))
ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1]); ax.set_ylabel("метрика")
ax.set_title("Фундаментальная проблема: у юзера два потенциальных исхода, реализуется один")
fig.tight_layout(); fig.savefig("v74_potential_outcomes.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 75. ATE / ATT / CATE
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 4.0))
xs = np.linspace(0, 10, 200)
eff = 20 + 8 * xs + 3 * np.sin(xs)
ax.plot(xs, eff, lw=2.8, color=C_MAIN, label="истинный эффект по сегментам CATE(x)")
ax.axhline(eff.mean(), color=C_ACC, lw=2.4, ls="--", label=f"ATE = {eff.mean():.0f} (среднее по всем)")
treated = xs > 6
ax.axhline(eff[treated].mean(), color=C_BAD, lw=2.4, ls=":",
           label=f"ATT = {eff[treated].mean():.0f} (для участвовавших — правый хвост)")
ax.fill_between(xs, eff, eff.mean(), alpha=0.08, color=C_MAIN)
ax.set_xlabel("сегмент / ковариата x (например, частота заказов)")
ax.set_ylabel("эффект воздействия")
ax.set_title("Один и тот же вопрос «в чём эффект?» — три разных ответа: ATE, ATT, CATE")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig("v75_ate_att_cate.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 76. DiD: параллельные тренды
# ============================================================
t = np.linspace(0, 10, 100)
ctrl = 100 + 2.2 * t
trt = 100 + 2.2 * t + 12 * (t > 5) * 1
ctrl_obs = ctrl + rng.normal(0, 1.2, 100) * 0.8
trt_obs = trt + rng.normal(0, 1.2, 100) * 0.8
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(t, ctrl_obs, "o", ms=3, color=C_MAIN, alpha=0.6)
ax.plot(t, trt_obs, "o", ms=3, color=C_ACC, alpha=0.6)
ax.plot(t[t < 5], ctrl[t < 5], color=C_MAIN, lw=2.6)
ax.plot(t[t >= 5], ctrl[t >= 5], color=C_MAIN, lw=2.6, ls="--", alpha=0.5)
ax.plot(t, trt, color=C_ACC, lw=2.6)
ax.axvline(5, color="gray", ls=":", lw=1.6)
ax.annotate("контрфакт теста:\nпродолжение тренда", xy=(7.5, 100 + 2.2 * 7.5), xytext=(6.0, 105),
            arrowprops=dict(arrowstyle="->", color=C_BAD), color=C_BAD, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
ax.annotate("", xy=(8.6, 100 + 2.2 * 8.6 + 12), xytext=(8.6, 100 + 2.2 * 8.6),
            arrowprops=dict(arrowstyle="<->", lw=2.6, color=C_TRUE))
ax.text(8.75, 100 + 2.2 * 8.6 + 5, "эффект DiD", color=C_TRUE, fontweight="bold", fontsize=11)
ax.text(2.5, 96, "до: тренды параллельны —\nключевое допущение", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#d8f3dc"))
ax.set_xlabel("время"); ax.set_ylabel("метрика")
ax.set_title("DiD: разность разностей — смотрим не уровни, а расхождение трендов")
fig.tight_layout(); fig.savefig("v76_did.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 77. Staggered TWFE: плохие контрали
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 3.9))
groups = [("Группа A (лечим в t1)", 1, C_TRUE), ("Группа B (лечим в t3)", 3, C_ACC), ("Группа C (лечим в t5)", 5, "#9467BD")]
t = np.arange(0, 8)
for name, gtime, c in groups:
    base = 100 + 2 * t
    eff = np.where(t >= gtime, 10, 0)
    ax.plot(t, base + eff, "o-", lw=2.4, color=c, label=name + (" (эффект +10)" ))
ax.axvspan(3, 5, color=C_BAD, alpha=0.08)
ax.text(4, 122, "окно t3–t5: группа A уже лечена.\nTWFE молча использует её как «контроль»\nдля C — и эффек C занижается",
        ha="center", fontsize=9, color=C_BAD,
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffe3e3"))
ax.set_xlabel("период t"); ax.set_ylabel("метрика")
ax.set_title("Ловушка staggered adoption: «ещё не лечёные» ≠ «никогда не лечёные»\n(лечение Callaway & Sant'Anna — сравнивать только с чистыми контрольными)")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("v77_staggered.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 78. RDD
# ============================================================
x = rng.uniform(40, 110, 400)
y0 = 20 + 0.5 * x + rng.normal(0, 4, 400)
y = y0 + np.where(x > 75, 9, 0)
fig, ax = plt.subplots(figsize=(8.6, 4.3))
ax.scatter(x, y, s=14, alpha=0.5, color="#8899aa")
xs1 = np.linspace(40, 75, 30); xs2 = np.linspace(75, 110, 30)
ax.plot(xs1, 20 + 0.5 * xs1, color=C_TRUE, lw=3)
ax.plot(xs2, 20 + 0.5 * xs2 + 9, color=C_BAD, lw=3)
ax.axvline(75, color="black", ls="--", lw=1.8)
ax.text(75.8, 74, "порог: 75 баллов\n(кредитный рейтинг)", fontsize=9)
ax.annotate("разрыв у порога = эффект\n(люди с 74 и 76 почти одинаковы)", xy=(75, 66), xytext=(45, 80),
            arrowprops=dict(arrowstyle="->", color="#333"), fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
ax.set_xlabel("running variable (балл)"); ax.set_ylabel("выручка юзера")
ax.set_title("RDD: рандомизацию делает сам порог — сравниваем «почти одинаковых» по обе стороны")
fig.tight_layout(); fig.savefig("v78_rdd.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 79. Synthetic control
# ============================================================
t = np.arange(0, 24)
treated_pre = 50 + 0.8 * t + 2 * np.sin(t / 2)
treated = treated_pre.copy()
treated[t >= 16] += 14 + 0.3 * (t[t >= 16] - 16)
synth = 50 + 0.8 * t + 2 * np.sin(t / 2) + rng.normal(0, 0.4, 24)
fig, ax = plt.subplots(figsize=(8.8, 4.3))
ax.plot(t, treated, "o-", lw=2.6, color=C_ACC, label="Екатеринбург (лечим в t=16)")
ax.plot(t, synth, "s--", lw=2.4, ms=5, color=C_MAIN, label="«синтетический Екб»: взвешенная смесь контрольных городов")
ax.fill_between(t[t >= 16], treated[t >= 16], synth[t >= 16], color=C_BAD, alpha=0.2)
ax.axvline(16, color="gray", ls=":", lw=1.6)
ax.text(8, 74, "до: смесь подогнана под тритмент\n(веса подбираются на претренде)", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#d8f3dc"))
ax.text(19.5, 62, "разрыв =\nэффект", color=C_BAD, fontweight="bold", fontsize=11)
ax.set_xlabel("месяц"); ax.set_ylabel("заказы")
ax.set_title("Synthetic control: контрфакт собираем из контрольных городов")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig("v79_synthetic_control.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 80. DiD + PSM: лестница оценщиков
# ============================================================
fig, ax = plt.subplots(figsize=(9.6, 4.2))
est = [35, 27, 21, 19.6, 20.1]
names = ["наивный pre-post", "DiD (сырые группы)", "+ ковариаты в регрессии", "PSM + DiD\n(отматченные)", "DR-DiD\n(Sant'Anna&Zhao)"]
cols = [C_BAD, C_BAD, C_ACC, C_MAIN, C_TRUE]
ax.bar(range(5), est, color=cols, width=0.6, alpha=0.9)
ax.axhline(20, color="black", ls="--", lw=2)
ax.text(4.45, 20.4, "истина = 20", fontsize=10, ha="right", fontweight="bold")
for i, v in enumerate(est):
    ax.text(i, v + 0.4, f"{v:.0f}", ha="center", fontweight="bold")
ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel("оценка эффекта")
ax.set_title("Лестница DiD+PSM (стиль LaLonde-теста): каждый шаг приближает к истине…\nа DR держится даже если одна из моделей врёт")
fig.tight_layout(); fig.savefig("v80_did_psm_ladder.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 81. Love plot: баланс до/после матчинга
# ============================================================
fig, ax = plt.subplots(figsize=(8.6, 4.4))
covs = ["возраст", "стаж", "частота заказов", "средний чек", "давность", "платформа iOS", "город МСК", "подписка", "время сессий", "число адресов"]
before = np.array([0.62, 0.48, 0.55, 0.31, 0.22, 0.18, 0.27, 0.12, 0.35, 0.08])
after = rng.uniform(0.01, 0.09, 10)
y = np.arange(len(covs))[::-1]
ax.scatter(before, y, s=90, color=C_BAD, label="до матчинга")
ax.scatter(after, y, s=90, color=C_TRUE, label="после матчинга")
for i in range(len(covs)):
    ax.plot([after[i], before[i]], [y[i], y[i]], color="#ccc", lw=1.2)
ax.axvline(0.1, color="black", ls="--", lw=1.8)
ax.text(0.105, 9.6, "порог SMD = 0.1", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(covs, fontsize=9)
ax.set_xlabel("стандартизированная разность средних (SMD)")
ax.set_title("Love plot: матчинг выровнял ковариаты — все точки ушли за порог 0.1")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig("v81_love_plot.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 82. IV-диаграмма
# ============================================================
fig, ax = plt.subplots(figsize=(9.2, 4.4))
ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
node(ax, 2, 8, "Z", C_ACC)   # инструмент
node(ax, 5, 8, "D")          # лечение
node(ax, 9, 8, "Y")          # исход
node(ax, 7, 2.5, "U", C_BAD) # конфаундер
edge(ax, 2.8, 8, 4.1, 8, C_ACC, lw=3)
edge(ax, 5.9, 8, 8.1, 8, lw=3)
edge(ax, 7.3, 3.6, 5.3, 7.1, C_BAD, lw=2.4)
edge(ax, 7.6, 3.6, 8.8, 7.1, C_BAD, lw=2.4)
ax.annotate("", xy=(9.6, 7.9), xytext=(2.2, 7.7),
            arrowprops=dict(arrowstyle="-", lw=2, color=C_BAD, linestyle=":"))
ax.text(5.9, 6.9, "запрещено: Z → Y напрямую\n(exclusion restriction)", fontsize=9.5, color=C_BAD,
        ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="#ffe3e3"))
ax.text(2, 9.6, "инструмент:\n«рандомная скидка-пуш»", ha="center", fontsize=9)
ax.text(5, 9.6, "лечение:\n«пользовался фичей»", ha="center", fontsize=9)
ax.text(9, 9.6, "исход:\nвыручка", ha="center", fontsize=9)
ax.text(7, 1.3, "ненаблюдаемый конфаундер\n(мотивация юзера)", ha="center", fontsize=9, color=C_BAD)
ax.set_title("IV: влияние Z на Y возможно только через D — тогда Wald покажет причинный эффект",
             fontsize=12, fontweight="bold")
fig.savefig("v82_iv.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 83. Карта методов: иерархия валидности
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
steps = [
    (0.5, "Рандомизация\n(A/B, switchback)", "допущений мало — но они есть (SUTVA, SRM)", C_TRUE, 4.0),
    (3.3, "Квазиэксперименты\nDiD / RDD / SC", "параллельные тренды, порог, претренд-веса", "#7FB069", 6.2),
    (6.4, "Матчинг / IPW\n+ ковариаты", "ignorability: все конфаундеры измерены", C_ACC, 8.4),
    (9.5, "Регрессия «как получится»\ncausal salad", "допущения неявные и хрупкие", C_BAD, 10.6),
]
for x, t1, t2, c, w in steps:
    ax.add_patch(plt.Rectangle((x, 3.2), 2.5, 3.4, fc=c, alpha=0.2, ec=c, lw=2.2))
    ax.text(x + 1.25, 5.6, t1, ha="center", fontsize=10, fontweight="bold")
    ax.text(x + 1.25, 3.9, t2, ha="center", fontsize=8, color="#444")
    ax.text(x + 1.25, 2.5, "▼", ha="center", fontsize=13, color="#999")
ax.text(6, 0.9, "вниз по лестнице: меньше веры в дизайн — больше веры в модель — меньше уверенности в результате",
        ha="center", fontsize=10, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd"))
ax.annotate("", xy=(9.4, 8.6), xytext=(0.6, 8.6),
            arrowprops=dict(arrowstyle="->", lw=2.6, color="#666"))
ax.text(5, 9.0, "спускайся только когда не пускают выше", ha="center", fontsize=10.5, fontweight="bold")
ax.set_title("Иерархия внутренней валидности: всегда спрашивай, почему тебе разрешили стоять выше",
             fontsize=12, fontweight="bold")
fig.savefig("v83_validity_ladder.png", bbox_inches="tight"); plt.close(fig)

print("Готово: v72–v83 (12 визуализаций) в", __file__.rsplit("/", 1)[0])
