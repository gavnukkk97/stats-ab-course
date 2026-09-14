# %% [markdown]
# # Визуализации для М7 «Проектирование АБ-платформы»
# Единый стиль: заголовок = вывод. Запуск: python3 make_visuals_m7.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
rng = np.random.default_rng(707)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_ACC, C_TRUE, C_BAD, C_GRY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C"

def box(ax, x, y, w, h, t, c="#F5F5F5", ec="#666", fs=9, bold=False):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")

def arrow(ax, x1, y1, x2, y2, c="#444", lw=2.2, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=lw, color=c, linestyle=ls))

# ============================================================
# 84. Референс-архитектура платформы (Microsoft ExP стиль)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
box(ax, 0.3, 5.6, 3.0, 1.6, "ПОРТАЛ\nдизайн-док, каталог метрик,\nкалендарь тестов", "#e8f0fe", fs=8.5, bold=True)
box(ax, 4.1, 5.6, 3.0, 1.6, "СПЛИТ-СЕРВИС\nдомены/слои/бакеты,\nsticky assignment", "#fff3cd", fs=8.5, bold=True)
box(ax, 7.9, 5.6, 3.0, 1.6, "ЛОГИРОВАНИЕ\nexposure + события,\nшина → озеро", "#e8f0fe", fs=8.5, bold=True)
box(ax, 11.7, 5.6, 2.0, 1.6, "АНАЛИЗ\nSRM-гейт,\nтесты, вердикт", "#d8f3dc", fs=8.5, bold=True)
arrow(ax, 3.3, 6.4, 4.1, 6.4); arrow(ax, 7.1, 6.4, 7.9, 6.4); arrow(ax, 10.9, 6.4, 11.7, 6.4)
box(ax, 4.1, 2.6, 3.0, 1.8, "Продукт\n(бэкенд/фронт):\n«в какую версию\nпопал юзер?»", "#fafafa", fs=8.5)
arrow(ax, 5.6, 4.4, 5.6, 5.6)
box(ax, 7.9, 2.6, 3.0, 1.8, "Озеро данных\nParquet, партиции\nпо дате/тесту", "#fafafa", fs=8.5)
arrow(ax, 9.4, 4.4, 9.4, 5.6)
box(ax, 0.3, 0.4, 5.5, 1.4, "Гейты качества: SRM-х², инварианты,\nguardrail авто-стоп (mSPRT)", "#ffe3e3", fs=8.5)
box(ax, 8.2, 0.4, 5.5, 1.4, "Выход: вердикт по decision rule,\nлента результатов, метрики velocity", "#d8f3dc", fs=8.5)
arrow(ax, 12.7, 5.6, 12.7, 1.8, c="#777", ls="--")
arrow(ax, 3.0, 1.8, 3.0, 5.6, c="#777", ls="--")
ax.set_title("Анатомия платформы экспериментов (по Microsoft ExP): 4 контура, гейты — не опция",
             fontsize=12, fontweight="bold")
fig.savefig("v84_platform_architecture.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 85. Домены → слои → бакеты (Google KDD'10)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
box(ax, 0.2, 6.4, 3.2, 1.2, "ДОМЕН «Поиск»", "#e8f0fe", bold=True, fs=9.5)
box(ax, 0.2, 4.6, 3.2, 1.2, "Слой: ранжирование\n(соль S1)", "#fff3cd", fs=8.5)
box(ax, 0.2, 2.8, 3.2, 1.2, "Слой: выдача\n(соль S2)", "#fff3cd", fs=8.5)
for i, (name, w) in enumerate([("тест A 40%", 0.4), ("тест B 20%", 0.2), ("ctrl", 0.4)]):
    box(ax, 4.0 + i * 3.3, 4.6, 3.0 * w * 3, 1.2, name, "#fafafa", fs=8)
ax.text(13.6, 5.2, "← бакеты 0–999:\nhash(user, S1) % 1000", ha="right", fontsize=8.5, color="#555")
arrow(ax, 1.8, 6.4, 1.8, 5.8); arrow(ax, 1.8, 4.6, 1.8, 4.0)
arrow(ax, 3.4, 5.2, 4.0, 5.2)
box(ax, 4.0, 2.8, 9.6, 1.2, "тут свои тесты — юзер участвует и в ранжировании, и в выдаче одновременно", "#fafafa", fs=8.5)
ax.text(7, 1.6, "Слой = независимая лотерея: f(user, соль_слоя) → бакет → вариант.\nЁмкость слоя = сумма долей тестов ≤ 100% (capacity planning)",
        ha="center", fontsize=9.5, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd"))
ax.set_title("Домены, слои и 1000 бакетов (Google, KDD 2010): параллельные тесты без пересечений",
             fontsize=12, fontweight="bold")
fig.savefig("v85_domains_layers.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 86. Exposure logging: честный timestamp
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4.0))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
box(ax, 0.3, 3.8, 2.6, 1.6, "Юзер видит\nновую кнопку\n(t=10:01)", "#fff3cd", fs=9)
box(ax, 3.5, 3.8, 2.6, 1.6, "Событие летит\nв шину\n(t=10:01:03)", "#fafafa", fs=9)
box(ax, 6.7, 3.8, 2.6, 1.6, "Батч в озеро\nночь → партиция\n(t=03:00+1д)", "#fafafa", fs=9)
arrow(ax, 2.9, 4.6, 3.5, 4.6); arrow(ax, 6.1, 4.6, 6.7, 4.6)
box(ax, 0.3, 1.0, 9.0, 1.9, "Правила: exposure = первый контакт (не лог-тайм!) · дедуп по (user, experiment) ·\nзапись до сбора метрик · потеря экспозиций = SRM", "#ffe3e3", fs=9)
ax.text(5, 0.4, "Uber: LRU-дедуп съедает 80% дублей exposure-логов — без него логи тонут в повторах",
        ha="center", fontsize=9, style="italic", color="#555")
ax.set_title("Exposure logging: точка входа юзера в тест — святое, всё остальное считается от неё",
             fontsize=12, fontweight="bold")
fig.savefig("v86_exposure_logging.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 87. Бакетизация: экономия вычислений
# ============================================================
rng2 = np.random.default_rng(7)
n_users = 1_000_000
per_user = rng2.lognormal(np.log(900), 1.0, n_users)
treat = per_user * 1.02
import time
from scipy import stats as st
t0 = time.perf_counter()
_ = st.ttest_ind(per_user[:500000], treat[500000:], equal_var=False)
t_full = time.perf_counter() - t0
for nb in [1000, 100]:
    idx = rng2.integers(0, nb, n_users)
    a = np.bincount(idx[:500000], weights=per_user[:500000], minlength=nb)
    b = np.bincount(idx[500000:], weights=treat[500000:], minlength=nb)
    cnt_a = np.bincount(idx[:500000], minlength=nb)
    cnt_b = np.bincount(idx[500000:], minlength=nb)
    t0 = time.perf_counter()
    _ = st.ttest_ind(a / np.maximum(cnt_a, 1), b / np.maximum(cnt_b, 1), equal_var=False)
    t_b = time.perf_counter() - t0
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.bar([0, 1, 2], [t_full * 1000, t_b * 1000 * 30, t_b * 1000], color=[C_BAD, C_MAIN, C_TRUE], width=0.55)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["поюзерно\n(1 млн строк)", "1000 бакетов", "100 бакетов"])
for i, v in enumerate([t_full * 1000, t_b * 1000 * 30, t_b * 1000]):
    ax.text(i, v * 1.05, f"{v:.1f} мс", ha="center", fontweight="bold")
ax.set_ylabel("время теста, мс (иллюстративно)")
ax.set_yscale("log")
ax.set_title("Бакетизация: миллион юзеров → сотни строк-агрегатов.\nМетрики платформы считаются на бакетах — это и тянет 500kk p-value в день (Avito)")
fig.tight_layout(); fig.savefig("v87_bucketization.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 88. Конвейер заявки: гейты качества
# ============================================================
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")
stages = [
    ("Заявка\n(design-док)", "#e8f0fe"),
    ("Валидация\nMDE→n, слой,\nконфликты", "#fff3cd"),
    ("Запуск\n+ramp 1→5→50%", "#fafafa"),
    ("ГЕЙТ: SRM χ²\n+ инварианты", "#ffe3e3"),
    ("ГЕЙТ: guardrails\nавто-стоп", "#ffe3e3"),
    ("Анализ +\ndecision rule", "#d8f3dc"),
    ("Вердикт\n→ лента", "#d8f3dc"),
]
for i, (t, c) in enumerate(stages):
    box(ax, 0.2 + i * 1.98, 2.6, 1.7, 2.0, t, c, fs=8)
    if i < len(stages) - 1:
        arrow(ax, 1.9 + i * 1.98, 3.6, 2.18 + i * 1.98, 3.6, lw=2)
ax.text(7, 1.2, "Zalando: без автоматического SRM-гейта SRM молчал в ≥20% тестов —\nгейты не «фича», а условие существования результатов",
        ha="center", fontsize=9.5, style="italic", color=C_BAD,
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffe3e3"))
ax.set_title("Путь гипотезы через платформу: два красных гейта, которые нельзя обойти",
             fontsize=12, fontweight="bold")
fig.savefig("v88_gates_pipeline.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 89. Календарь гипотез (gantt)
# ============================================================
fig, ax = plt.subplots(figsize=(10.5, 4.6))
layers = ["Слой: доставка", "Слой: поиск", "Слой: промо"]
tests = [
    (0, 0, 3, "T1", C_MAIN), (0, 3, 2, "T4", C_ACC), (0, 5, 3, "T7", C_TRUE),
    (1, 0, 2, "T2", C_ACC), (1, 2, 4, "T5", C_MAIN),
    (2, 0, 4, "T3", C_TRUE), (2, 4, 2, "T6", C_MAIN),
]
for li, start, dur, name, c in tests:
    ax.barh(li, dur, left=start, height=0.55, color=c, alpha=0.75, edgecolor="white")
    ax.text(start + dur / 2, li, f"{name}\n{dur} нед", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
ax.set_yticks(range(3)); ax.set_yticklabels(layers)
ax.set_xlabel("недели"); ax.set_xlim(0, 9)
ax.axvline(2.3, color=C_BAD, ls="--", lw=1.8)
ax.text(2.4, 2.62, "новая гипотеза:\nближайший слот — слой «промо», неделя 4\n(в «поиске» — очередь до недели 6)", fontsize=8.5, color=C_BAD,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff3cd"))
ax.set_title("Платформа-календарь: слои × время — система сама находит ближайший свободный слот под MDE гипотезы")
fig.tight_layout(); fig.savefig("v89_calendar.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 90. FIFO vs экономика слота
# ============================================================
rng3 = np.random.default_rng(70)
value = lambda: rng3.uniform(0.5, 2.0)
fifo_total, econ_total, n_sim = 0, 0, 2000
for _ in range(n_sim):
    hyp = [(value(), rng3.uniform(1, 4)) for _ in range(12)]
    cap = 3.0
    fifo = sorted(hyp, key=lambda h: -h[1])[:6]
    fifo_total += sum(v for v, d in fifo if sum(dd for _, dd in fifo[:fifo.index((v, d)) + 1]) <= cap)
    econ = sorted(hyp, key=lambda h: -h[0] / h[1])
    tot, acc = 0, 0
    for v, d in econ:
        if acc + d <= cap:
            acc += d; tot += v
    econ_total += tot
fig, ax = plt.subplots(figsize=(8.2, 4.2))
ax.bar([0, 1], [fifo_total / n_sim, econ_total / n_sim], color=[C_GRY, C_TRUE], width=0.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["«Кто первый» (FIFO)", "Экономика слота:\nmax ценность/слот-день"])
for i, v in enumerate([fifo_total / n_sim, econ_total / n_sim]):
    ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold")
ax.set_ylabel("реализованная ценность за окно (среднее)")
ax.set_title(f"Календарь с экономикой слота реализует на ~{(econ_total/fifo_total-1)*100:.0f}% больше ценности\nпри том же трафике — очередь это тоже продукт")
fig.tight_layout(); fig.savefig("v90_slot_economics.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 91. Velocity: где умирают гипотезы
# ============================================================
fig, ax = plt.subplots(figsize=(10, 3.8))
stages_v = [("Идея в бэклоге", 100, C_GRY), ("Дизайн-док\nнаписан", 70, C_MAIN),
            ("Прошёл валидацию\nплатформы", 55, C_MAIN), ("Слот получен", 40, C_ACC),
            ("Тест завершён", 33, C_ACC), ("Вердикт → решение", 30, C_TRUE)]
for i, (t, v, c) in enumerate(stages_v):
    ax.bar(i, v, color=c, width=0.6, alpha=0.9)
    ax.text(i, v + 2, f"{v}", ha="center", fontweight="bold", fontsize=9)
ax.set_xticks(range(len(stages_v))); ax.set_xticklabels([s[0] for s in stages_v], fontsize=8)
ax.set_ylabel("из 100 идей")
ax.annotate("вакуум №1: ожидание слота\n(календарь из 7.4)", xy=(3, 40), xytext=(1.2, 70),
            arrowprops=dict(arrowstyle="->", color=C_BAD), color=C_BAD, fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
ax.set_title("Velocity-воронка: время идея→решение — метрика платформы, а не только команд")
fig.tight_layout(); fig.savefig("v91_velocity.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 92. Buy vs build
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
box(ax, 0.3, 4.6, 5.4, 2.6, "BUILD (своя)\n+ полная интеграция с твоими данными\n+ календарь/экономика под себя (ниша!)\n− годы команды, поддержка 24/7", "#e8f0fe", fs=9)
box(ax, 6.3, 4.6, 5.4, 2.6, "BUY (Statsig/Eppo/GrowthBook)\n+ запустить за недели\n+ проверенная статистика\n− свои данные/метрик-хаб не влезают,\n− календаря с экономикой нет ни у кого", "#fff3cd", fs=9)
box(ax, 0.3, 1.2, 11.4, 2.4, "Прагматичный путь (кейс Avito/trisigma): ядро — своё (сплит+данные+календарь),\nстатистика — проверенные библиотеки, UX — как у вендоров; платформа окупается\nскоростью решений, а не лицензией", "#d8f3dc", fs=9.5)
ax.set_title("Buy vs build: спор не «что дешевле», а «где наша уникальность» (trisigma: 150 млн ₽ vs 2,5 млн ₽/год)",
             fontsize=11.5, fontweight="bold")
fig.savefig("v92_buy_build.png", bbox_inches="tight"); plt.close(fig)

print("Готово: v84–v92 (9 визуализаций) в", __file__.rsplit("/", 1)[0])
