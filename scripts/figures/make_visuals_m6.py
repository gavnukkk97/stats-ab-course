from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "course" / "modules" / "M6" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Визуализации для М6 «Каузальный вывод»
# Единый стиль: заголовок = вывод. Запуск: python3 scripts/figures/make_visuals_m6.py
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
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v72_dag_structures.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 73. Симпсон с DAG
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
# Counts deliberately differ between arms: every stratum worsens but the aggregate improves.
seg = [("высокая базовая конверсия", 100, 20, 900, 162),
       ("низкая базовая конверсия", 900, 72, 100, 6)]
for i, (name, nc, yc, nt, yt) in enumerate(seg):
    ca, cb = yc / nc, yt / nt
    assert cb < ca
    axes[0].bar([i-.18, i+.18], [100*ca, 100*cb], color=[C_TRUE,C_BAD], width=.32)
    for xpos, rate, total in [(i-.18, ca, nc), (i+.18, cb, nt)]:
        axes[0].text(xpos, rate*100+.3, f"{rate*100:.0f}%\nn={total}", ha="center", fontsize=9)
axes[0].set_xticks(range(2)); axes[0].set_xticklabels([x[0].replace(" базовая", "\nбазовая") for x in seg])
axes[0].set_ylabel("конверсия, %"); axes[0].set_ylim(0,25)
axes[0].set_title("В каждом сегменте тест хуже контроля", fontsize=10)
axes[0].legend([mpatches.Patch(color=C_TRUE), mpatches.Patch(color=C_BAD)], ["контроль","тест"], fontsize=8)
agg_a = sum(s[2] for s in seg)/sum(s[1] for s in seg)
agg_b = sum(s[4] for s in seg)/sum(s[3] for s in seg)
assert agg_b > agg_a
axes[0].text(.5, -.24, f"Агрегат: тест {agg_b:.1%} > контроль {agg_a:.1%}", transform=axes[0].transAxes,
             ha="center", fontsize=10, color=C_BAD, fontweight="bold")
ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
node(ax, 5, 8.5, "P", C_ACC); node(ax, 2.5, 3.5, "D"); node(ax, 7.5, 3.5, "Y")
edge(ax, 4.5, 7.8, 3.1, 4.3, C_ACC); edge(ax, 5.5, 7.8, 6.9, 4.3, C_ACC)
edge(ax, 3.3, 3.5, 6.7, 3.5)
ax.text(5, 6.3, "P = платформа\n(конфаундер: и дизайн,\nи конверсия зависят от неё)",
        ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
ax.text(5, 1.5, "D = новый дизайн, Y = конверсия", ha="center", fontsize=9, color="#555")
fig.suptitle("Парадокс Симпсона: агрегат «выигрывает», потому что тест попал на богатую платформу",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v73_simpson_dag.png", bbox_inches="tight"); plt.close(fig)

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
for world, values, observed in [(0,y0,~obs_t),(1,y1,obs_t)]:
    ax.scatter(np.full(n, world), values, s=55, facecolors="white", edgecolors=C_MAIN, zorder=2)
    ax.scatter(np.full(observed.sum(), world), values[observed], s=55, color=C_MAIN, zorder=3)
ax.text(0, 185, "мир БЕЗ воздействия\nY(0)", ha="center", fontsize=11, fontweight="bold")
ax.text(1, 185, "мир С воздействием\nY(1)", ha="center", fontsize=11, fontweight="bold")
ax.text(0.5, 62, "эффект каждого = разность его миров\n(наблюдаем только один — закрашенный)",
        ha="center", fontsize=10, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd"))
ax.set_ylim(76, 197); ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1]); ax.set_ylabel("метрика")
ax.set_title("Фундаментальная проблема: у юзера два потенциальных исхода, реализуется один")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v74_potential_outcomes.png", bbox_inches="tight"); plt.close(fig)

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
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v75_ate_att_cate.png", bbox_inches="tight"); plt.close(fig)

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
ax.text(2.5, 96, "До: диагностируем тренды;\nв post допущение не проверить", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#d8f3dc"))
ax.set_xlabel("время"); ax.set_ylabel("метрика")
ax.set_title("DiD: разность разностей — смотрим не уровни, а расхождение трендов")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v76_did.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 77. Staggered TWFE: неподходящие сравнения
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 3.9))
groups = [("A: старт t=1",1,C_TRUE),("B: старт t=3",3,C_ACC),("C: старт t=5",5,"#9467BD")]
t = np.arange(8)
dmat, ymat, effects = [], [], []
for name,gtime,c in groups:
    d = (t>=gtime).astype(float)
    eff = d*(10+5*np.maximum(t-gtime,0))
    y = 100+2*t+eff
    dmat.append(d); ymat.append(y); effects.append(eff)
    ax.plot(t,y,"o-",lw=2,color=c,label=name)
dmat,ymat,effects=map(np.array,(dmat,ymat,effects))
dres=dmat-dmat.mean(axis=1,keepdims=True)-dmat.mean(axis=0,keepdims=True)+dmat.mean()
yres=ymat-ymat.mean(axis=1,keepdims=True)-ymat.mean(axis=0,keepdims=True)+ymat.mean()
twfe=(dres*yres).sum()/(dres*dres).sum(); att=effects[dmat==1].mean()
ax.text(.03,.96,f"Эффект растёт с длительностью: 10 + 5×стаж\nTWFE = {twfe:.1f}; средний эффект по treated-периодам = {att:.1f}",
        transform=ax.transAxes,va="top",fontsize=9,bbox=dict(fc="white",alpha=.85))
ax.set_xlabel("период t"); ax.set_ylabel("метрика")
ax.set_title("При динамических эффектах TWFE может смешивать несовместимые сравнения")
ax.legend(fontsize=9,loc="lower right")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v77_staggered.png", bbox_inches="tight"); plt.close(fig)

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
ax.annotate("Разрыв = локальный эффект\nпри непрерывности Y(0) и Y(1)", xy=(75, 66), xytext=(45, 80),
            arrowprops=dict(arrowstyle="->", color="#333"), fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
ax.set_xlabel("running variable (балл)"); ax.set_ylabel("выручка юзера")
ax.set_title("RDD: порог даёт дизайн; причинный вывод требует допущений")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v78_rdd.png", bbox_inches="tight"); plt.close(fig)

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
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v79_synthetic_control.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 80. DiD + PSM: лестница оценщиков
# ============================================================
fig, ax = plt.subplots(figsize=(10,4.3)); ax.axis("off")
rows = [
 ["DiD", "Параллельные тренды без X", "ATT выбранной группы и периода"],
 ["Regression / matching + DiD", "Условные тренды по X; overlap", "ATT; при отсечении — другая группа"],
 ["DR-DiD", "Условные тренды; overlap; верна одна nuisance-модель", "Заданный ATT, при регулярности"],
]
table=ax.table(cellText=rows,colLabels=["Оценка", "Что нужно обосновать", "Что сравниваем"],cellLoc="left",loc="center",colWidths=[.24,.44,.32])
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,3.6)
ax.set_title("Корректировка не гарантирует улучшение: сравнивайте допущения и estimand",pad=20)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v80_did_psm_ladder.png",bbox_inches="tight"); plt.close(fig)

# ============================================================
# 81. Love plot: баланс до/после матчинга
# ============================================================
fig, ax = plt.subplots(figsize=(8.6, 4.4))
covs = ["возраст", "стаж", "частота заказов", "средний чек", "давность", "платформа iOS", "город МСК", "прежняя подписка", "время сессий до", "число адресов"]
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
ax.set_title("Иллюстрация Love plot: |SMD| < 0.1 — ориентир, не доказательство exchangeability")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "v81_love_plot.png", bbox_inches="tight"); plt.close(fig)

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
ax.text(2, 9.6, "инструмент:\n«случайное приглашение»", ha="center", fontsize=9)
ax.text(5, 9.6, "лечение:\n«пользовался фичей»", ha="center", fontsize=9)
ax.text(9, 9.6, "исход:\nвыручка", ha="center", fontsize=9)
ax.text(7, 1.3, "ненаблюдаемый конфаундер\n(мотивация юзера)", ha="center", fontsize=9, color=C_BAD)
ax.set_title("IV: relevance + independence + exclusion + monotonicity → LATE",
             fontsize=12, fontweight="bold")
fig.savefig(OUTPUT_DIR / "v82_iv.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 83. Карта методов: иерархия валидности
# ============================================================
fig,ax=plt.subplots(figsize=(11,5)); ax.axis("off")
rows=[
 ["Рандомизация", "Назначение, соблюдение протокола, интерференция", "ATE / ITT в целевой популяции"],
 ["DiD", "Контроль + pre; контрфактические параллельные тренды", "ATT по группе и периоду"],
 ["RDD / IV", "Порог и непрерывность / допустимый инструмент", "Локальный эффект / LATE"],
 ["Synthetic control", "Незатронутые доноры + pre-fit + перенос на post", "Эффект у treated-юнита"],
 ["Matching / IPW / DR", "Достаточные pre-X, exchangeability, overlap", "ATE / ATT / эффект в overlap"],
 ["ITS", "Стабильная экстраполяция; нет одновременных шоков", "Эффект у исследуемого ряда"],
]
table=ax.table(cellText=rows,colLabels=["Дизайн", "Ключевые допущения", "Целевая величина"],cellLoc="left",loc="center",colWidths=[.2,.53,.27])
table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,2.6)
ax.set_title("Выбирайте идентификацию под вопрос: универсального ранга доверия нет",pad=18)
fig.tight_layout();fig.savefig(OUTPUT_DIR / "v83_validity_ladder.png",bbox_inches="tight");plt.close(fig)

print("Готово: v72–v83 (12 визуализаций) в", OUTPUT_DIR)
