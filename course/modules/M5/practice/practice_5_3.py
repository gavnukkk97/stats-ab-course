# -*- coding: utf-8 -*-
"""Практика 5.3 — Байесовский A/B: P(B>A), expected loss, когда останавливаться
(сквозной кейс «ЕдаДома»).

Не верь — проверяй симуляцией.

Что делаем:
1) ПОЛНЫЙ БАЙЕСОВСКИЙ A/B С НУЛЯ (numpy, без MCMC): конверсии -> Beta-апостериоры
   по группам -> Монте-Карло сэмплы -> P(B>A), expected loss, ROPE-критерий.
   MC сверяем с замкнутой формулой P(B>A) и численным интегрированием EL;
2) СРАВНЕНИЕ С ЧАСТОТНЫМ на 5 сценариях (крупный эффект / мелкий / нет
   эффекта / мало данных / спекулятивный априор): p-value против P(B>A),
   EL и ROPE — где выводы совпадают, где расходятся и почему;
3) ЧУВСТВИТЕЛЬНОСТЬ К АПРИОРУ: спекулятивный Beta(300,700) против flat
   при росте n — приор «весит» свои (a+b) псевдонаблюдений;
4) БАЙЕС И ПОДГЛЯДЫВАНИЕ — честная картинка: A/A-тест с ежедневной
   проверкой. «Остановиться, когда P(B>A)>95%» при ежедневных взглядах
   ловит ложные победы чаще 5% (байес НЕ лечит подглядывание, если мерить
   частотно — Robinson);
5) ПРАВИЛО EXPECTED LOSS («threshold of caring», Stucchio) на трёх истинах
   (B лучше +0,6 п.п. / равны / B хуже −0,6 п.п.): оно оптимизирует РЕШЕНИЕ,
   а не знание: останавливается одинаково рано и при эффекте, и без него;
   контролирует СРЕДНЮЮ потерю (порог держится), но в отдельных мирах
   раскатывает худший вариант с потерей в разы выше порога.

Запуск из корня репозитория: python3 course/modules/M5/practice/practice_5_3.py
Графики сохраняются в папку images/ этого модуля.
"""

# %% [markdown]
# # Практика 5.3 — байесовский A/B на конверсиях «ЕдаДома»
# Экран оплаты: контроль A (старый) и B (новый). Метрика — конверсия
# визит->заказ ~10%. Всё считается БЕЗ MCMC: сопряжённость Beta-Бернулли
# даёт апостериор аналитически, Монте-Карло нужен только для разности.

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betaln

RNG = np.random.default_rng(53)  # seed -> результат воспроизводим
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05
EPS = 0.001          # threshold of caring: 0.1 п.п. конверсии (бизнес-порог)
ROPE = 0.001         # ROPE: разница меньше 0.1 п.п. — практический нуль
MC = 200_000         # сэмплов Монте-Карло для одного анализа


# %%
def posterior(visits, orders, prior=(1.0, 1.0)):
    """Beta-апостериор конверсии: prior + данные (сопряжённость)."""
    a, b = prior
    return a + orders, b + visits - orders


def p_b_better_mc(sa, sb):
    """P(B>A) Монте-Карло: в скольких мирах сэмпл B выше сэмпла A."""
    return float((sb > sa).mean())


def expected_loss_mc(sa, sb):
    """EL катим B = E[max(A-B, 0)] Монте-Карло: сколько теряем, если
    раскатим B, а на самом деле лучше A."""
    return float(np.maximum(sa - sb, 0.0).mean())


def p_b_better_exact(aa, ba, ab, bb):
    """Замкнутая формула P(B>A) для целочисленных бета-параметров
    (Evan Miller / Stucchio): проверка MC вообще без сэмплов."""
    aa, ba, ab, bb = (np.float64(x) for x in (aa, ba, ab, bb))
    total = 0.0
    for i in np.arange(ab):
        num = betaln(aa + i, ba + bb)
        den = np.log(bb + i) + betaln(1 + i, bb) + betaln(aa, ba)
        total += np.exp(num - den)
    return total


def expected_loss_grid(aa, ba, ab, bb):
    """EL численным интегрированием без MC:
    E[max(A-B,0)] = E[A*F_B(A)] - E[B*(1-F_A(B))] (два 1-D интеграла)."""
    m = aa / (aa + ba)
    s = np.sqrt(aa * ba / (aa + ba) ** 2 / (aa + ba + 1))
    g = np.linspace(max(1e-9, m - 12 * s), min(1 - 1e-9, m + 12 * s), 400_001)
    t1 = np.trapezoid(g * stats.beta.cdf(g, ab, bb) * stats.beta.pdf(g, aa, ba), g)
    t2 = np.trapezoid(g * stats.beta.sf(g, aa, ba) * stats.beta.pdf(g, ab, bb), g)
    return t1 - t2


def hdi(samples, cred=0.95):
    """Самый короткий интервал, накрывающий cred массы (Kruschke)."""
    s = np.sort(samples)
    n = len(s)
    k = int(np.floor(cred * n))
    widths = s[k:] - s[: n - k]        # интервалы [s[j], s[j+k]]
    i = int(np.argmin(widths))
    return s[i], s[i + k]


def rope_verdict(d_samples, rope=ROPE, cred=0.95):
    """ROPE-критерий: HDI разности целиком внутри ROPE -> эквивалентны;
    целиком вне -> различие подтверждено; пересекает -> данных мало."""
    lo, hi = hdi(d_samples, cred)
    if lo > rope:
        return "B лучше: эффект выше ROPE", lo, hi
    if hi < -rope:
        return "B хуже: эффект ниже ROPE", lo, hi
    if hi < rope and lo > -rope:
        return "практически эквивалентны", lo, hi
    return "неопределённость — данных мало", lo, hi


def freq_two_prop(cA, nA, cB, nB):
    """Частотный z-тест двух долей (как в М3)."""
    pA, pB = cA / nA, cB / nB
    p = (cA + cB) / (nA + nB)
    se = np.sqrt(p * (1 - p) * (1 / nA + 1 / nB))
    z = (pB - pA) / se
    return float(2 * stats.norm.sf(abs(z)))


# %% [markdown]
# ## Шаг 1. Полный пайплайн на одном тесте (главный пример урока)
# Новый экран оплаты: 20 000 визитов, сплит 50/50.
# A: 10 000 визитов, 1 012 заказов; B: 10 000 визитов, 1 063 заказа.
# Приор Beta(1,1) (flat): до данных все конверсии равновозможны.

# %%
nA, cA = 10_000, 1_012
nB, cB = 10_000, 1_063
aA, bA = posterior(nA, cA)          # Beta(1013, 8989)
aB, bB = posterior(nB, cB)          # Beta(1064, 8938)

sa = stats.beta.rvs(aA, bA, size=MC, random_state=RNG)
sb = stats.beta.rvs(aB, bB, size=MC, random_state=RNG)
d = sb - sa

pba = p_b_better_mc(sa, sb)
el_b = expected_loss_mc(sa, sb)
el_a = expected_loss_mc(sb, sa)     # E[max(B-A,0)]: потери, если оставим A
pba_exact = p_b_better_exact(aA, bA, aB, bB)
el_grid = expected_loss_grid(aA, bA, aB, bB)
verdict, lo, hi = rope_verdict(d)
p_b3 = float((sb > 1.03 * sa).mean())   # P(B лучше A МИНИМУМ на 3% (отн.)
pv1 = freq_two_prop(cA, nA, cB, nB)

print("=" * 88)
print("1) Новый экран оплаты: A = {}/{} ({:.2f}%),  B = {}/{} ({:.2f}%)".format(
    cA, nA, 100 * cA / nA, cB, nB, 100 * cB / nB))
print(f"   апостериоры: A ~ Beta({aA:.0f}, {bA:.0f}),  B ~ Beta({aB:.0f}, {bB:.0f})")
print(f"   P(B>A)         Монте-Карло ({MC:,} сэмплов):        {pba:.4f}")
print(f"   P(B>A)         замкнутая формула (Miller/Stucchio): {pba_exact:.4f}")
print(f"   P(B > 1.03*A)  «B лучше минимум на 3% (относит.)»:  {p_b3:.4f}")
print(f"   expected loss, катим B:  E[max(A-B,0)] MC: {el_b * 100:.4f} п.п."
      f"   (числ. интегр.: {el_grid * 100:.4f} п.п.)")
print(f"   expected loss, оставляем A: E[max(B-A,0)]: {el_a * 100:.4f} п.п.")
print(f"   95% HDI разности (B-A): [{lo * 100:+.3f}; {hi * 100:+.3f}] п.п."
      f"  -> ROPE ±{ROPE * 100:.1f} п.п.: {verdict}")
print()
print("   ТРИ ЯЗЫКА ОДНИХ ДАННЫХ:")
print(f"   частотный: p = {pv1:.3f} -> {'значимо' if pv1 < ALPHA else 'НЕ значимо (серый)'};"
      " (гипотезу о равенстве не отвергаем)")
print(f"   байес о превосходстве: P(B>A) = {pba:.0%} -> "
      f"{'порог 95% пройден' if pba > 0.95 else 'порог 95% НЕ пройден'}")
print(f"   байес о решении (Stucchio): EL(B) = {el_b * 100:.2f} п.п. < порога заботы"
      f" {EPS * 100:.1f} п.п. -> КАТИМ B: апостериорная ожидаемая потеря"
      " лишь 0,02 п.п.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

grid = np.linspace(0.085, 0.12, 600)
ax = axes[0]
ax.plot(grid, stats.beta.pdf(grid, aA, bA), lw=2.4, color="#4C72B0", label="A: старый экран")
ax.plot(grid, stats.beta.pdf(grid, aB, bB), lw=2.4, color="#DD8452", label="B: новый экран")
ax.fill_between(grid, stats.beta.pdf(grid, aA, bA), color="#4C72B0", alpha=0.10)
ax.fill_between(grid, stats.beta.pdf(grid, aB, bB), color="#DD8452", alpha=0.10)
ax.set_title(f"Апостериоры конверсии: P(B>A) = {pba:.2f} — «в {pba:.0%} миров B лучше»")
ax.set_xlabel("конверсия")
ax.set_ylabel("плотность апостериора")
ax.legend()

ax = axes[1]
ax.hist(d * 100, bins=90, color="#55A868", alpha=0.85)
ax.axvline(0, color="k", lw=1)
ax.axvspan(-ROPE * 100, ROPE * 100, color="#C44E52", alpha=0.25,
           label=f"ROPE ±{ROPE * 100:.1f} п.п.")
ax.axvline(EPS * 100, color="C3", ls="--", lw=1.5, label="порог заботы 0,1 п.п.")
ax.axvline(lo * 100, color="k", ls=":", lw=1.5)
ax.axvline(hi * 100, color="k", ls=":", lw=1.5,
           label="края 95% HDI")
ax.set_title(f"Разность B-A: хвост слева от 0 = EL(B) = {el_b * 100:.2f} п.п. < 0,1"
             " -> катить можно")
ax.set_xlabel("разность конверсий, п.п.")
ax.set_ylabel("число MC-сэмпла")
ax.legend()
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_5_3_posteriors.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Шаг 2. Байес против частотного: 5 сценариев
# Наблюдения задаём детерминированно (round(n*p)), чтобы выводы сценариев
# были стабильны. Пороги: p<0,05; P(B>A)>95%; EL<0,1 п.п.
# Сценарий 5 — те же данные, что 4, но «эксперт обещал конверсию ~30%»:
# априор Beta(300,700) на руку B.

# %%
SCENARIOS = [
    dict(name="1. крупный эффект",        nA=5_000, cA=500, nB=5_000, cB=650,
         priorB=(1, 1)),
    dict(name="2. мелкий эффект",         nA=20_000, cA=2_000, nB=20_000, cB=2_120,
         priorB=(1, 1)),
    dict(name="3. нет эффекта, n=1 млн",  nA=1_000_000, cA=100_000, nB=1_000_000,
         cB=100_000, priorB=(1, 1)),
    dict(name="4. мало данных",           nA=200, cA=20, nB=200, cB=26,
         priorB=(1, 1)),
    dict(name="5. мало данных + априор 30%", nA=200, cA=20, nB=200, cB=26,
         priorB=(300, 700)),
]

rows = []
for sc in SCENARIOS:
    aA_, bA_ = posterior(sc["nA"], sc["cA"])
    aB_, bB_ = posterior(sc["nB"], sc["cB"], prior=sc["priorB"])
    sa_ = stats.beta.rvs(aA_, bA_, size=50_000, random_state=RNG)
    sb_ = stats.beta.rvs(aB_, bB_, size=50_000, random_state=RNG)
    d_ = sb_ - sa_
    pv = freq_two_prop(sc["cA"], sc["nA"], sc["cB"], sc["nB"])
    verdict, _, _ = rope_verdict(d_)
    elb = expected_loss_mc(sa_, sb_)
    rows.append(dict(
        сценарий=sc["name"],
        B_набл=f"{100 * sc['cB'] / sc['nB']:.2f}%",
        p_value=round(pv, 4),
        p_val_решение=("зелёный" if pv < ALPHA else "серый"),
        P_B_A=round(p_b_better_mc(sa_, sb_), 4),
        EL_B_пп=round(elb * 100, 3),
        EL_решение=("катим" if elb < EPS else "ждём"),
        ROPE=verdict,
    ))
tab = pd.DataFrame(rows)

print("=" * 88)
print("2) Частотный против байесовского, 5 сценариев "
      "(порог: p<0,05; P(B>A)>95%; EL<0,1 п.п.)")
print(tab.to_string(index=False))
print(
    "\n   Читаем таблицу:\n"
    "   - сценарий 2: p=0,049 еле «зелёный», а байес говорит то же самое, но\n"
    "     В ЕДИНИЦАХ ПОТЕРЬ: EL=0,003 п.п. — значимость тут ни при чём;\n"
    "   - сценарий 3: p=1,0 «не отвергаем H0» (языка «принять H0» нет),\n"
    "     а ROPE уверенно говорит «практически эквивалентны» — но для этого\n"
    "     понадобился МИЛЛИОН наблюдений на руку: доказать эквивалентность\n"
    "     дороже, чем найти разницу;\n"
    "   - сценарий 4: все три метода честно молчат — данных мало;\n"
    "   - сценарий 5: p-value всё ещё серый (данные те же!), а P(B>A)=1,000\n"
    "     и EL=0 -> «катим». Расхождение создал АПРИОР, не данные:\n"
    "     1000 псевдонаблюдений «эксперта» задавили 200 реальных."
)

# %%
fig, ax = plt.subplots(figsize=(10.0, 4.8))
xs = np.arange(len(tab))
ax.bar(xs - 0.2, tab["p_value"], width=0.4, color="#4C72B0", label="p-value (порог 0,05)")
ax.bar(xs + 0.2, 1 - tab["P_B_A"], width=0.4, color="#DD8452",
       label="1 − P(B>A) (аналогичный порог 0,05)")
ax.axhline(ALPHA, color="C3", ls="--", lw=1.5, label="порог решения 5%")
ax.set_yscale("log")
ax.set_xticks(xs, tab["сценарий"], rotation=12)
ax.set_ylabel("p-value и 1−P(B>A), log-шкала")
ax.set_title("p-value ≈ 1−P(B>A) при больших n и честном априоре (сценарии 1–4);"
             "\nсценарий 5 расходится драматически — виноват априор, не данные")
for x, pv in zip(xs, tab["p_value"]):
    ax.annotate(f"{pv:.3f}", (x - 0.2, max(pv, 1e-4)), ha="center", va="bottom", fontsize=8)
for x, pb in zip(xs, tab["P_B_A"]):
    ax.annotate(f"{1 - pb:.3f}", (x + 0.2, max(1 - pb, 1e-4)), ha="center", va="bottom",
                fontsize=8)
ax.legend()
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_5_3_scenarios.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Шаг 3. Чувствительность к априору: тот же спекулятивный Beta(300,700),
# но данных всё больше. Приор «весит» ровно свои (a+b) наблюдений.

# %%
rows2 = []
for n in (200, 1_000, 5_000, 20_000):
    c = int(round(n * 0.13))
    for prior, tag in [((1, 1), "flat Beta(1,1)"),
                       ((300, 700), "спекулятивный Beta(300,700)")]:
        aB_, bB_ = posterior(n, c, prior=prior)
        sb_ = stats.beta.rvs(aB_, bB_, size=50_000, random_state=RNG)
        rows2.append(dict(
            n_B=n,
            априор=tag,
            P_B_A=round(p_b_better_mc(sa[:50_000], sb_), 3),
            медиана_B=round(float(stats.beta.ppf(0.5, aB_, bB_)), 4),
        ))
tab2 = pd.DataFrame(rows2)
print("=" * 88)
print("3) Чувствительность: спекулятивный априор против flat при росте n "
      "(истина: конверсия B = 13%)")
print(tab2.to_string(index=False))
print("   при n=200 медиана B со спекулятивным априором 0,272 против истины 0,13 —\n"
      "   приор задавил данные; к n=20 000 данные побеждают (0,138 против 0,130 у\n"
      "   flat): приор «весит» свои (a+b)=1000 псевдонаблюдений, дальше тает.")

# %% [markdown]
# ## Шаг 4. Байес и подглядывание: честная картинка
# A/A-тест (эффекта НЕТ), 2000 юзеров/день на руку, 28 дней, 20 000 миров.
# Правила остановки на ОДНИХ И ТЕХ ЖЕ данных:
#  (а) частотный, 1 взгляд на 28-й день: односторонний p<0,05 -> альфа = 5%;
#  (б) частотный, ежедневный взгляд: стоп при одностороннем p<0,05 (урок 3.7);
#  (в) байес «P(B>A)>95%»: 1 взгляд на 28-й день;
#  (г) байес «P(B>A)>95%»: ежедневный взгляд;
#  (д) expected loss (Stucchio): стоп при min(EL_A, EL_B) < 0,1 п.п.
# Для скорости — нормальное приближение бета-апостериоров (при n>=2000
# на руку совпадает с точным MC с точностью ~0,001).

# %%
N_DAY = 2_000
DAYS = 28
WORLDS = 20_000
K = np.arange(1, DAYS + 1)
NK = N_DAY * K                       # юзеров на руку к дню k
ROWS = np.arange(WORLDS)


def daily_paths(rng, pA, pB):
    """Дневные P(B>A), EL_A, EL_B, p-value по накопленным данным."""
    conv_A = rng.binomial(N_DAY, pA, size=(WORLDS, DAYS)).cumsum(axis=1)
    conv_B = rng.binomial(N_DAY, pB, size=(WORLDS, DAYS)).cumsum(axis=1)
    n_k = np.broadcast_to(NK, conv_A.shape)
    mA = (conv_A + 1) / (n_k + 2)                       # средние Beta(1+x, 1+n-x)
    mB = (conv_B + 1) / (n_k + 2)
    vA = mA * (1 - mA) / (n_k + 3)
    vB = mB * (1 - mB) / (n_k + 3)
    sd = np.sqrt(vA + vB)
    zq = (mB - mA) / sd
    p_ba = stats.norm.cdf(zq)                            # P(B>A)
    # EL катим B: A-B ~ N(-mu, sd^2), mu=mB-mA:
    # E[max(A-B,0)] = sd*phi(mu/sd) - mu*Phi(-mu/sd)
    el_B = sd * stats.norm.pdf(zq) - (mB - mA) * stats.norm.cdf(-zq)
    el_A = sd * stats.norm.pdf(zq) + (mB - mA) * stats.norm.cdf(zq)
    p_pool = (conv_A + conv_B) / (2 * n_k)
    se = np.sqrt(p_pool * (1 - p_pool) * 2 / n_k)
    p_val = stats.norm.sf((conv_B - conv_A) / (se * n_k))  # одностороннее B>A
    return p_ba, el_A, el_B, p_val


p_ba0, elA0, elB0, pval0 = daily_paths(RNG, 0.10, 0.10)   # A/A: эффекта нет

hit_p = pval0 < ALPHA
hit_b = p_ba0 > 0.95
stop0 = np.minimum(elA0, elB0) < EPS
day0 = np.where(stop0.any(axis=1), stop0.argmax(axis=1) + 1, DAYS + 1)
idx0 = np.clip(day0 - 1, 0, DAYS - 1)
vict_at_stop0 = p_ba0[ROWS, idx0] > 0.95

print("=" * 88)
print(f"4) A/A-тест (эффекта НЕТ), {N_DAY:,} юзеров/день на руку, {WORLDS:,} миров")
print(f"   (а) частотный, ОДИН взгляд (день 28), p<0,05:      ложных побед "
      f"{hit_p[:, 27].mean():.1%}  (номинал 5%)")
print(f"   (б) частотный, ежедневный взгляд, стоп при p<0,05:  ложных побед "
      f"{hit_p.any(axis=1).mean():.1%}  <- урок 3.7")
print(f"   (в) байес, ОДИН взгляд (день 28), P(B>A)>95%:       ложных побед "
      f"{hit_b[:, 27].mean():.1%}  (совпадает с (а): 95% <-> альфа 5%)")
print(f"   (г) байес, ЕЖЕДНЕВНЫЙ взгляд, стоп при P(B>A)>95%:  ложных побед "
      f"{hit_b.any(axis=1).mean():.1%}  <- байес НЕ лечит подглядывание")
print(f"   (д) expected loss, стоп при min(EL)<{EPS * 100:.1f} п.п.: медианный день"
      f" остановки {np.median(day0):.0f}; остановились с «P(B>A)>95%» — "
      f"{vict_at_stop0.mean():.1%} миров")
print("       EL-порог — НЕ альфа: он не про частоту побед, а про цену решения.")

# %% [markdown]
# ### Шаг 5. Правило EL на трёх истинах: что именно оно контролирует
# Одна и та же процедура (ежедневная проверка, стоп при min(EL)<0,1 п.п.,
# катим вариант с меньшим EL), но истинный эффект разный; это диагностика, не доказательство uniform bound:
# +0,6 п.п. (B лучше) / 0 (A/A) / −0,6 п.п. (B хуже).
# Считаем: медианный день решения; долю «раскатали B»; среднюю РЕАЛИЗОВАННУЮ
# потерю решения (если раскатали худший — теряем истинную разницу навсегда).

# %%
HARMS = [("+0,6 п.п. (B лучше)", 0.100, 0.106),
         (" 0 (A/A)", 0.100, 0.100),
         ("−0,6 п.п. (B хуже)", 0.100, 0.094)]
truth_rows, loss_harm = [], None
for tag, pA, pB in HARMS:
    pba_, elA_, elB_, _ = daily_paths(RNG, pA, pB)
    stop_ = np.minimum(elA_, elB_) < EPS
    day_ = np.where(stop_.any(axis=1), stop_.argmax(axis=1) + 1, DAYS + 1)
    idx_ = np.clip(day_ - 1, 0, DAYS - 1)
    roll_b = elB_[ROWS, idx_] <= elA_[ROWS, idx_]      # катим того, у кого EL ниже
    chosen = np.where(roll_b, pB, pA)
    loss = np.maximum(max(pA, pB) - chosen, 0.0)       # реализованная потеря
    truth_rows.append(dict(
        истина=tag,
        медианный_день=int(np.median(day_)),
        раскатали_B=f"{roll_b.mean():.0%}",
        ошибок_раскатки=f"{(loss > 0).mean():.0%}",
        средняя_потеря_пп=f"{loss.mean() * 100:.3f}",
        порог_пп=f"{EPS * 100:.1f}",
    ))
    if pB < pA:
        loss_harm = loss

tab3 = pd.DataFrame(truth_rows)
print("=" * 88)
print(f"5) Правило expected loss (Stucchio) на трёх истинах, {WORLDS:,} миров,"
      " ежедневные проверки")
print(tab3.to_string(index=False))
share_wrong = (loss_harm > 0).mean()
print(
    f"\n   ИСТИНА −0,6 п.п.: в {share_wrong:.0%} миров правило раскатало ХУДШИЙ вариант\n"
    f"   (реализованная потеря 0,6 п.п. = {0.6 / (EPS * 100):.0f}x порога в каждом таком мире),\n"
    f"   но СРЕДНЯЯ потеря {loss_harm.mean() * 100:.3f} п.п. ниже порога {EPS * 100:.1f} п.п.:\n"
    "   EL — апостериорная ожидаемая потеря при заданной модели и приоре —\n"
    "   Эта симуляция трёх истин не доказывает частотную гарантию риска для всех эффектов."
)

# %%
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

ax = axes[0]
rules = [
    ("частотный\n1 взгляд", hit_p[:, 27].mean(), "#4C72B0"),
    ("частотный\nежедневно", hit_p.any(axis=1).mean(), "#4C72B0"),
    ("байес P>95%\n1 взгляд", hit_b[:, 27].mean(), "#DD8452"),
    ("байес P>95%\nежедневно", hit_b.any(axis=1).mean(), "#DD8452"),
]
bars = ax.bar([r[0] for r in rules], [r[1] * 100 for r in rules],
              color=[r[2] for r in rules], alpha=0.9)
ax.axhline(5, color="C3", ls="--", lw=1.5, label="номинал 5%")
for bar, (_, v, _) in zip(bars, rules):
    ax.annotate(f"{v:.1%}", (bar.get_x() + bar.get_width() / 2, v * 100),
                ha="center", va="bottom")
ax.set_ylabel("доля ложных побед в A/A, %")
ax.tick_params(axis="x", labelsize=8)
ax.set_title("Байес НЕ лечит подглядывание: «стоп при P(B>A)>95%»\nс ежедневными проверками ловит ложные победы впятеро чаще 5%")
ax.legend(fontsize=8)

ax = axes[1]
for pba, tag, color in [(p_ba0, "A/A (эффекта нет)", "#4C72B0")]:
    med = np.median(pba, axis=0)
    q10 = np.quantile(pba, 0.10, axis=0)
    q90 = np.quantile(pba, 0.90, axis=0)
    ax.plot(K, med, lw=2.4, color=color, label=tag + ", медиана")
    ax.fill_between(K, q10, q90, color=color, alpha=0.15, label="10–90% миров")
ax.axhline(0.95, color="C3", ls="--", lw=1.2, label="порог P(B>A)=95%")
ax.axhline(0.5, color="grey", ls=":", lw=1)
ax.set_ylim(0.05, 1.0)
ax.set_xlabel("день теста")
ax.set_ylabel("P(B>A) по накопленным данным")
ax.set_title("Апостериор пересчитывать можно хоть каждый час\n(likelihood principle) — но частота пересечения порога и есть цена подглядывания")
ax.legend(loc="upper left", fontsize=8)

ax = axes[2]
ax.hist(np.where(loss_harm > 0, loss_harm, 0) * 100,
        bins=np.linspace(-0.02, 0.65, 40), color="#C44E52", alpha=0.85)
ax.axvline(EPS * 100, color="k", ls="--", lw=1.8,
           label=f"порог заботы {EPS * 100:.1f} п.п.")
ax.axvline(loss_harm.mean() * 100, color="#4C72B0", lw=2,
           label=f"средняя потеря {loss_harm.mean() * 100:.2f} п.п.")
ax.set_xlabel("реализованная потеря решения, п.п. (истина: B хуже на 0,6)")
ax.set_ylabel("число миров")
ax.set_title("EL держит СРЕДНЮЮ потерю ниже порога, но отдельные миры\n"
             f"теряют 0,6 п.п. ({share_wrong:.0%} миров) — контроль не по каждому миру")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_5_3_peeking.png", dpi=150)
plt.close(fig)

print("\nГрафики: practice_5_3_posteriors.png, practice_5_3_scenarios.png, "
      "practice_5_3_peeking.png")


# %% [markdown]
# ## Шаг 6. Operating characteristics: эффекты × priors
# Нормальная аппроксимация Beta posterior, как в daily_paths; это диагностика,
# не доказательство uniform bound. Сильный ошибочный prior даёт контрпример.

# %%
def el_policy_risk(effect, prior_b=(1., 1.), worlds=2000, days=28, seed=5306):
    rng = np.random.default_rng(seed)
    active = np.ones(worlds, dtype=bool)
    choose_b = np.zeros(worlds, dtype=bool)
    stop_day = np.full(worlds, days)
    ca = np.zeros(worlds)
    cb = np.zeros(worlds)
    satisfied = np.zeros(worlds, dtype=bool)
    for day in range(1, days+1):
        ca += rng.binomial(2000, .10, worlds)
        cb += rng.binomial(2000, .10+effect, worlds)
        n = day*2000
        a1, a2 = ca+1, n-ca+1
        b1, b2 = cb+prior_b[0], n-cb+prior_b[1]
        mu = b1/(b1+b2)-a1/(a1+a2)
        var = a1*a2/((a1+a2)**2*(a1+a2+1)) + b1*b2/((b1+b2)**2*(b1+b2+1))
        sd = np.sqrt(var)
        el_b = sd*stats.norm.pdf(mu/sd)-mu*stats.norm.cdf(-mu/sd)
        el_a = sd*stats.norm.pdf(mu/sd)+mu*stats.norm.cdf(mu/sd)
        hit = np.minimum(el_a, el_b) < EPS
        stop = active & (hit | (day == days))
        choose_b[stop] = el_b[stop] < el_a[stop]
        stop_day[stop] = day
        satisfied[stop] = hit[stop]
        active[stop] = False
        if not active.any():
            break
    loss = np.where(choose_b, max(-effect, 0), max(effect, 0))
    return float(loss.mean()), float(choose_b.mean()), float(satisfied.mean()), float(np.median(stop_day))

risk_rows = []
for prior in [(1., 1.), (10., 90.), (300., 700.)]:
    for effect in [-.02, -.01, -.006, -.003, 0., .003, .006, .01, .02]:
        risk, share, met, median = el_policy_risk(effect, prior)
        risk_rows.append(dict(prior_B=str(prior), effect_pp=100*effect,
                              mean_realized_loss_pp=100*risk, choose_B=share,
                              threshold_met=met, median_day=median))
print("6) Сетка effects × priors: frequentist loss при фиксированной истине (не posterior EL)")
print(pd.DataFrame(risk_rows).round(4).to_string(index=False))
print("   Ошибочный сильный prior может остановить тест с posterior EL<epsilon,")
print("   но realized risk больше epsilon. Последний день без порога — вынужденное решение.")
