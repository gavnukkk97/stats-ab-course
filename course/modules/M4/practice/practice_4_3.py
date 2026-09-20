# -*- coding: utf-8 -*-
"""Практика 4.3 — Валидность и ловушки АБ-теста (кейс «ЕдаДома»).

Не верь — проверь симуляцией. Четыре ловушки, которые ломают тест ещё до
того, как p-value успеет что-то сказать:

1) парадокс Симпсона: в агрегате тест «выигрывает», внутри обоих сегментов —
   проигрывает (сдвиг состава групп);
2) интерференция на двухстороннем рынке: спилловор-доля γ — поюзерный сплит
   видит лишь (1−γ) истинного эффекта, гео-дизайн — весь эффект;
3) эффект новизны/обучения: траектория эффекта по неделям и то, что
   «увидит» тест разной длительности;
4) чек-лист валидности как python-функция — прогоняем тест
   «умная очередь курьеров» и смотрим, что не пускает его в анализ.

Запуск из корня репозитория: python3 course/modules/M4/practice/practice_4_3.py
Графики сохраняются в папку images/ этого модуля.
"""

# %% [markdown]
# # Практика 4.3 — Валидность: четыре способа обмануться
# Правило урока: сначала валидность — потом p-value. Пока чек-лист не зелёный,
# «прокрас» ничего не значит.

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(43)  # seed -> результат воспроизводим
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05


def prop_z(k1, n1, k2, n2):
    """Двухвыборочный z-тест долей (пуленная дисперсия)."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return p1, p2, z, 2 * stats.norm.sf(abs(z))


# %% [markdown]
# ## Шаг 1. Парадокс Симпсона: агрегат «выиграл», сегменты «проиграли»
# Тест «стикеры-пожелания к заказу». Фича раскатана через НОВЫЙ клиент
# приложения, поэтому в triggered-аудитории теста мобильных 85% (в контроле
# они попадаются через старый клиент — 50%). Мобильные конвертятся в 2 раза
# лучше web. Внутри ОБЕИХ платформ тест чуть хуже контроля — но в агрегате
# «выигрывает», потому что в нём больше конверсионных мобильных.

# %%
N_ARM = 50_000
# платформа: (доля в контроле, конверсия контроля, доля в тесте, конверсия теста)
PLAN = {
    "mobile": (0.50, 0.120, 0.85, 0.114),
    "web": (0.50, 0.060, 0.15, 0.053),
}

rows, cells = [], {}
for plat, (sc, cc, st, ct) in PLAN.items():
    n_c = RNG.binomial(N_ARM, sc)
    n_t = RNG.binomial(N_ARM, st)
    k_c = RNG.binomial(n_c, cc)
    k_t = RNG.binomial(n_t, ct)
    p_t, p_c, z, p = prop_z(k_t, n_t, k_c, n_c)
    cells[plat] = (n_c, k_c, n_t, k_t)
    rows.append({
        "сегмент": plat,
        "n ctrl": n_c, "conv ctrl": p_c,
        "n test": n_t, "conv test": p_t,
        "разница, п.п.": (p_t - p_c) * 100,
        "z": z, "p-value": p,
        "вердикт": "тест ХУЖЕ (значимо)" if p < ALPHA and z < 0 else
                   ("тест лучше (значимо)" if p < ALPHA else "нет разницы"),
    })

kc, nc = sum(v[1] for v in cells.values()), sum(v[0] for v in cells.values())
kt, nt = sum(v[3] for v in cells.values()), sum(v[2] for v in cells.values())
pt, pc, zg, pg = prop_z(kt, nt, kc, nc)
rows.append({
    "сегмент": "АГРЕГАТ", "n ctrl": nc, "conv ctrl": pc, "n test": nt,
    "conv test": pt, "разница, п.п.": (pt - pc) * 100, "z": zg, "p-value": pg,
    "вердикт": "тест ЛУЧШЕ (значимо)" if pg < ALPHA and zg > 0 else "нет разницы",
})
tab1 = pd.DataFrame(rows).set_index("сегмент")

chi2, p_comp, _, _ = stats.chi2_contingency([
    [cells["mobile"][0], cells["mobile"][2]],
    [cells["web"][0], cells["web"][2]],
])

print("=" * 96)
print(f"1) Симпсон: конверсия в первый заказ, n = {N_ARM:,} на вариант".replace(",", " "))
print(tab1.round(4).to_string())
print(f"   Состав платформ по вариантам различается (мобильных: 50% ctrl vs 85% test):")
print(f"   «сегментный SRM» — chi2 = {chi2:,.0f}, p < 1e-300 — составы групп НЕ сравнимы.".replace(",", " "))
print("   Агрегатная «победа» ~ +1 п.п. — артефакт состава; внутри обеих платформ фича хуже.")

# %%
fig, ax = plt.subplots(figsize=(9.2, 4.4))
labels = ["web", "mobile", "АГРЕГАТ"]
xc = [tab1.loc[l, "conv ctrl"] * 100 for l in labels]
xt = [tab1.loc[l, "conv test"] * 100 for l in labels]
x = np.arange(3)
ax.bar(x - 0.19, xc, 0.38, color="#4C72B0", label="контроль")
ax.bar(x + 0.19, xt, 0.38, color="#C44E52", label="тест")
for i, l in enumerate(labels):
    d = tab1.loc[l, "разница, п.п."]
    win = d > 0
    ax.annotate(f"{d:+.1f} п.п. {'(тест «выиграл»)' if win else '(тест проиграл)'}",
                (i, max(xc[i], xt[i]) + 0.25), ha="center", fontsize=10,
                fontweight="bold", color="#55A868" if win else "#C44E52")
ax.set_xticks(x, labels)
ax.set_ylabel("конверсия в первый заказ, %")
ax.set_title("Парадокс Симпсона: тест хуже в обоих сегментах,\nно «выигрывает» в агрегате — состав групп разный")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_4_3_simpson.png", dpi=150)
print("Сохранено: practice_4_3_simpson.png")

# %% [markdown]
# ## Шаг 2. Интерференция: двухсторонний рынок делит курьеров
# Фича «умная очередь»: полная раскатка реально ускоряет доставку на
# δ = −4 мин. Механика спилловора: доля γ выигрыша тестовых юзеров
# реализуется через ОБЩИЙ пул курьеров и разливается на всех (контрольных —
# в том числе). Проверяем модель: поюзерный сплит должен видеть (1−γ)·δ,
# гео-дизайн (город целиком) — всё δ. И заодно: ловит ли A/A интерференцию.

# %%
TRUE_DELTA = -4.0        # мин; истинный эффект полной раскатки
BASE, SD, N_PER, REPS = 38.0, 10.0, 2_000, 400
P_TEST = 0.5


def world_split(gamma, delta, rng):
    """Поюзерный сплит 50/50: доля gamma эффекта уходит в общий пул."""
    n = 2 * N_PER
    T = rng.integers(0, 2, n)
    pool = gamma * delta * P_TEST                       # разлилось на ВСЕХ
    shift = np.where(T == 1, (1 - gamma) * delta + pool, pool)
    return BASE + shift + rng.normal(0, SD, n), T


rows = []
for gamma in (0.0, 0.3, 0.6, 0.9):
    ests, hits = [], 0
    for _ in range(REPS):
        M, T = world_split(gamma, TRUE_DELTA, RNG)
        t, p = stats.ttest_ind(M[T == 1], M[T == 0], equal_var=False)
        ests.append(M[T == 1].mean() - M[T == 0].mean())
        hits += p < ALPHA
    ests = np.array(ests)
    rows.append({
        "γ (утечка)": gamma,
        "видит поюзерный сплит, мин": ests.mean(),
        "теория (1−γ)·δ": (1 - gamma) * TRUE_DELTA,
        "доля истинного эффекта": ests.mean() / TRUE_DELTA,
        "мощность теста": hits / REPS,
    })
tab2 = pd.DataFrame(rows).set_index("γ (утечка)")

# гео-дизайн: город целиком в тесте -> спилловор остаётся внутри кластера
geo_ests = []
for _ in range(REPS):
    m_c = BASE + RNG.normal(0, SD / np.sqrt(N_PER))
    m_t = BASE + TRUE_DELTA + RNG.normal(0, SD / np.sqrt(N_PER))
    geo_ests.append(m_t - m_c)
geo_ests = np.array(geo_ests)

# A/A при выключенной фиче: интерференция молчит (δ=0 -> пул пустой)
aa_hits = 0
for _ in range(REPS):
    M, T = world_split(0.9, 0.0, RNG)
    aa_hits += stats.ttest_ind(M[T == 1], M[T == 0], equal_var=False).pvalue < ALPHA

print("=" * 96)
print(f"2) Спилловор на рынке доставки (истинный эффект полной раскатки δ = {TRUE_DELTA} мин, {REPS} миров)")
print(tab2.round(3).to_string())
print(f"   Гео-дизайн (город целиком): видит {geo_ests.mean():.2f} мин — весь эффект при любом γ.")
print(f"   A/A-тест при γ=0.9 и выключенной фиче: ложных срабатываний {aa_hits / REPS:.1%}.")
print("   -> A/A НЕ ловит интерференцию: утечка проявляется только когда фича включена и имеет эффект.")
print("   -> при γ=0.9 поюзерный тест видит ~10% эффекта: рабочую фичу хоронят в 4 мирах из 5.")

# %%
fig, ax = plt.subplots(figsize=(8.6, 4.3))
gam = np.linspace(0, 0.95, 100)
ax.plot(gam, 1 - gam, lw=2.4, color="#4C72B0", label="теория: поюзерный сплит видит (1−γ)")
ax.plot(tab2.index, tab2["доля истинного эффекта"], "o", ms=11, color="#C44E52",
        label="симуляция: измерено/истина")
ax.axhline(1.0, color="#55A868", ls="--", lw=2, label="гео-дизайн: видит весь эффект")
ax.annotate("γ=0.9: видно ~10% эффекта,\nмощность падает до ~20%", (0.9, 0.10),
            xytext=(-175, 30), textcoords="offset points", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#C44E52"))
ax.set_xlabel("доля эффекта, утекающая в общий пул курьеров, γ")
ax.set_ylabel("какую долю истинного эффекта видит тест")
ax.set_title("Интерференция размазывает эффект на контроль:\nпоюзерный сплит слеп, гео-дизайн — нет")
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_4_3_spillover.png", dpi=150)
print("Сохранено: practice_4_3_spillover.png")

# %% [markdown]
# ## Шаг 3. Новизна и обучение: что видит тест разной длительности
# Стационарный эффект +1,0%. Новизна: старт вздёрнут на A п.п. и затухает
# (τ — скорость). Обучение (learning): старт «в минус», юзеры привыкают.
# Тест длительностью W недель видит СРЕДНЕЕ по первым W неделям —
# и это среднее может отличаться от стационара в разы.

# %%
WEEKS, DELTA_INF, TAU = 6, 1.0, 0.7
A_NOV, A_LEARN = 2.0, -1.5  # п.п. добавки в 1-ю неделю

w = np.arange(1, WEEKS + 1)
nov_week = DELTA_INF + A_NOV * np.exp(-(w - 1) / TAU)
lrn_week = DELTA_INF + A_LEARN * np.exp(-(w - 1) / TAU)
nov_cum = np.cumsum(nov_week) / w
lrn_cum = np.cumsum(lrn_week) / w

tab3 = pd.DataFrame({
    "новизна: эффект недели, %": nov_week,
    "новизна: видит тест W недель, %": nov_cum,
    "обучение: эффект недели, %": lrn_week,
    "обучение: видит тест W недель, %": lrn_cum,
}, index=[f"неделя {i}" for i in w])

print("=" * 96)
print(f"3) Траектории эффекта (стационар = +{DELTA_INF}%, τ = {TAU} нед.)")
print(tab3.round(2).to_string())
print("   Новизна: тест в 1 неделю отчитает +3,0% (вздёрнуто в 3 раза); даже 6 недель — +1,5%,")
print("   остаток новизны не выветрился. Решение по короткому тесту переоценивает долгий эффект.")
print("   Обучение: тест в 1 неделю видит −0,5% и ХОРОНИТ рабочую фичу (вердикт HOLD вместо ROLL);")
print("   2 недели — «нет эффекта»; верить можно только длинному окну/кумулятивной кривой.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), sharey=True)
for ax, week, cum, ttl in (
    (axes[0], nov_week, nov_cum, "Эффект новизны: вздёрнутый старт -> затухание"),
    (axes[1], lrn_week, lrn_cum, "Learning effect: сначала хуже -> обучение -> рост"),
):
    ax.bar(w, week, 0.6, color="#4C72B0", alpha=0.75, label="эффект недели")
    ax.plot(w, cum, "o-", color="#C44E52", lw=2.2, label="что видит тест W недель")
    ax.axhline(DELTA_INF, color="black", ls="--", lw=1.6, label="стационарный эффект +1,0%")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("неделя теста")
    ax.set_title(ttl, fontsize=10.5)
    ax.set_xticks(w)
    ax.legend(fontsize=8.5)
axes[0].set_ylabel("эффект, п.п. к ARPU")
fig.suptitle("Окно теста решает, что вы увидите: короткий тест меряет траекторию, а не стационар",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_4_3_novelty.png", dpi=150)
print("Сохранено: practice_4_3_novelty.png")

# %% [markdown]
# ## Шаг 4. Чек-лист валидности как python-функция
# Прогоняем тест «умная очередь курьеров»: сплит честный, A/A свежий, а вот
# юниты связаны общими курьерами, окно короткое, id — cookie. Смотрим, какие
# пункты горят и что делать с каждым.

# %%
def validity_checklist(cfg):
    """Чек-лист валидности ПЕРЕД чтением результатов.

    Возвращает список (пункт, статус, действие). Статусы:
    OK / ПРОВЕРИТЬ (жёлтый) / СТОП (красный — p-value читать нельзя).
    """
    res = []

    def add(name, status, action):
        res.append((name, status, action))

    add("SRM: сплит соответствует плану", "OK" if cfg["srm_p"] > 0.001 else "СТОП",
        f"chi2 p = {cfg['srm_p']:.3f}" if cfg["srm_p"] > 0.001 else
        f"SRM! p = {cfg['srm_p']:.1e}: дебаг по чек-листу урока 4.2, результатам верить нельзя")
    add("A/A: платформа свежая (последний A/A < квартала назад)", "OK" if cfg["aa_fresh"] else "ПРОВЕРИТЬ",
        "p-value равномерны" if cfg["aa_fresh"] else "поставить A/A параллельно и сверить уровень")
    add("Юнит рандомизации = юнит анализа", "OK" if cfg["unit_match"] else "СТОП",
        "user -> user (3.2)" if cfg["unit_match"] else "пересчитать на юните назначения, иначе alpha сломана")
    interf = cfg["interference"]
    add("Интерференция юнитов (SUTVA)", {"none": "OK", "market": "СТОП", "social": "СТОП"}.get(interf, "СТОП"),
        "юниты независимы" if interf == "none" else
        "юниты делят ресурс -> кластерный/гео-дизайн или switchback (гл.22 Kohavi)")
    overlap = cfg["overlap_pct"]
    add("Пересечение с другими тестами", "OK" if overlap < 1 else "ПРОВЕРИТЬ",
        f"пересечение {overlap}% — пренебрежимо" if overlap < 1 else
        f"пересечение {overlap}%: проверка 2x2 взаимодействие (koch-kir), при конфликте — разнести тесты")
    dur = cfg["duration_weeks"]
    add("Длительность: целые недели, не меньше 2, сезонный цикл покрыт", "OK" if dur >= 2 else "СТОП",
        f"{dur} нед.: ok" if dur >= 2 else f"{dur} нед.: продлить минимум до 2 (будни/выходные + новизна)")
    add("Траектория эффекта стабильна (новизна/обучение)", "OK" if cfg["novelty_flat"] else "ПРОВЕРИТЬ",
        "недельные эффекты на плато" if cfg["novelty_flat"] else
        "эффект ещё плывёт: смотреть кумулятив, решение — по длинному окну или holdout (4.5)")
    add("Идентификатор юнита стабильный", "OK" if cfg["id_type"] == "account" else "ПРОВЕРИТЬ",
        "account id" if cfg["id_type"] == "account" else
        "cookie churn: юзеры прыгают между вариантами, эффект дробится (Deng гл.7) — перейти на account id")
    trig = cfg["trigger_rule"]
    add("Популяция анализа: ITT либо обоснованный безопасный trigger", "OK" if trig == "itt" else ("ПРОВЕРИТЬ" if trig == "triggered_even" else "СТОП"),
        "анализируем всех рандомизированных (ITT)" if trig == "itt" else
        "одинаковое правило недостаточно: проверить отсутствие зависимости eligibility от treatment" if trig == "triggered_even" else
        "фильтр «видел фичу» асимметричен -> survivorship/Симпсон: вернуть ITT")
    add("Телеметрия: проверки полноты и механизма потерь", "OK" if cfg["telemetry_ok"] else "СТОП",
        "полнота подтверждена отдельной диагностикой; продуктовые события могут меняться" if cfg["telemetry_ok"] else
        "потери событий различаются по вариантам: тест невалиден, чинить логирование")
    add("Защитные метрики в допустимых пределах (краши, время ответа)", "OK" if cfg["guardrails_alive"] else "СТОП",
        "в норме" if cfg["guardrails_alive"] else "платформа деградирует — сначала чинить, потом мерить")
    return res


CFG_COURIER = {
    "srm_p": 0.62, "aa_fresh": True, "unit_match": True, "interference": "market",
    "overlap_pct": 3.0, "duration_weeks": 1, "novelty_flat": False,
    "id_type": "cookie", "trigger_rule": "itt", "telemetry_ok": True, "guardrails_alive": True,
}

checks = validity_checklist(CFG_COURIER)
print("=" * 96)
print("4) Чек-лист валидности: тест «умная очередь курьеров»")
for name, status, action in checks:
    print(f"   [{status:^10}] {name}")
    print(f"                -> {action}")
n_stop = sum(s == "СТОП" for _, s, _ in checks)
n_warn = sum(s == "ПРОВЕРИТЬ" for _, s, _ in checks)
print(f"   Итог: {n_stop} СТОП, {n_warn} ПРОВЕРИТЬ из {len(checks)} пунктов.")
print("   Пока есть СТОП — p-value не читаем: сначала гео/switchback-дизайн, окно >= 2 недель, account id.")

# %%
print()
print("=" * 96)
print("ГЛАВНОЕ (что должны увидеть):")
print(f"1) Симпсон: в web и mobile тест хуже на "
      f"{abs(tab1.loc['web', 'разница, п.п.']):.1f} и {abs(tab1.loc['mobile', 'разница, п.п.']):.1f} п.п.,")
print(f"   в агрегате «лучше» на {tab1.loc['АГРЕГАТ', 'разница, п.п.']:+.1f} п.п. "
      f"(p={tab1.loc['АГРЕГАТ', 'p-value']:.1e}) — состав групп разъехал.")
print("2) Интерференция: поюзерный сплит видит (1−γ)·δ; при γ=0.9 — десятую часть эффекта")
print(f"   и мощность {tab2.loc[0.9, 'мощность теста']:.0%}; гео-дизайн видит всё ({geo_ests.mean():.1f} мин); A/A интерференцию не ловит.")
print(f"3) Новизна: 1 неделя — +{nov_cum[0]:.1f}% (в 3 раза выше стационара), 6 недель — +{nov_cum[-1]:.1f}%.")
print(f"   Обучение: 1 неделя — {lrn_cum[0]:.1f}% (рабочую фичу хороним), 6 недель — +{lrn_cum[-1]:.1f}%.")
print(f"4) Чек-лист: {n_stop} СТОП у «умной очереди» — до их исправления результат не читают.")
