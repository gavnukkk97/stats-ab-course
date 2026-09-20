# -*- coding: utf-8 -*-
"""Практика 5.2 — Интерпретация: credible vs confidence, вероятности по-человечески.

Принцип курса: не верь формуле — проверь симуляцией.

Что делаем:
1) одна задача (120 заказов на 1000 визитов): частотный доверительный интервал
   (Вальд, Вильсон) против байесовского credible при разных априорах —
   при плоском априоре числа практически совпадают, при информативном расходятся;
2) base rate fallacy: медицинский тест (чувствительность 95%, специфичность 90%)
   через естественные частоты «на 10 000 человек» + кривая P(болен|+) от
   превалентности; бизнес-прокси: антифрод «ЕдаДома»;
3) «вероятность, что B лучше A»: Монте-Карло по двум Beta-апостериорам против
   p-value той же пары — словарь перевода + чувствительность к скептическому априору.

Запуск из корня репозитория: python3 course/modules/M5/practice/practice_5_2.py
Графики сохраняются в папку images/ этого модуля.
"""

# %% [markdown]
# Шаг 0. Импорты. Кейс: баннер «Завтраки за 199 руб» — конверсия в заказ.

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260913)
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

Z95 = stats.norm.ppf(0.975)


def wald_interval(k, n):
    """Вальд (обычный) ДИ для доли."""
    p = k / n
    se = np.sqrt(p * (1 - p) / n)
    return p - Z95 * se, p + Z95 * se


def wilson_interval(k, n):
    """ДИ Вильсона для доли (частотный, честный при малых n и редких событиях)."""
    p, z = k / n, Z95
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def credible(a, b):
    """Равносторонний 95% credible-интервал Beta(a, b)."""
    return stats.beta.ppf(0.025, a, b), stats.beta.ppf(0.975, a, b)


def ztest_two_props(k_a, n_a, k_b, n_b):
    """Двусторонний z-тест двух долей (пулленная дисперсия): z и p."""
    p_pool = (k_a + k_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (k_b / n_b - k_a / n_a) / se
    p2 = 2 * stats.norm.sf(abs(z))
    return z, p2, p2 / 2  # p2/2 = одностороннее p при положительном эффекте


# %% [markdown]
# Шаг 1. CI против credible: 120 заказов на 1000 визитов карточки (12.0%).

# %%
K, N = 120, 1000
p_hat = K / N
wald = wald_interval(K, N)
wilson = wilson_interval(K, N)
variants = [
    ("Вальд, частотный CI", wald),
    ("Вильсон, частотный CI", wilson),
    ("credible, априор Beta(1,1) плоский", credible(K + 1, N - K + 1)),
    ("credible, априор Beta(2,2) слабый", credible(K + 2, N - K + 2)),
    ("credible, априор Beta(8,92) исторический", credible(K + 8, N - K + 92)),
]
print("=" * 78)
print(f"1) CI vs CREDIBLE: {K} заказов на {N} визитов (MLE = {p_hat:.1%})")
print("-" * 78)
for name, (lo, hi) in variants:
    print(f"   {name:<42}: [{lo:.4f}; {hi:.4f}]  = [{lo:.2%}; {hi:.2%}]")
d_wilson = np.abs(np.array(wilson) - np.array(credible(K + 1, N - K + 1)))
print(f"\n   |Вильсон - credible(плоский)| = {d_wilson.max():.4%} — на практике одно и то же число.")
print(f"   Но СМЫСЛ разный: у CI гарантия «в 95% повторений накроет»,")
print(f"   у credible — «с вероятностью 95% параметр здесь» (см. урок).")

# %%
fig, ax = plt.subplots(figsize=(11.2, 4.0))
for i, (name, (lo, hi)) in enumerate(variants):
    color = "#4C72B0" if i < 2 else ("#55A868" if i == 2 else "#DD8452")
    ax.plot([lo, hi], [i, i], lw=5, color=color, alpha=0.85, marker="|", ms=12)
    ax.text(hi + 0.003, i, f"[{lo:.2%}; {hi:.2%}]", va="center", fontsize=8.5)
ax.axvline(p_hat, color="k", ls=":", lw=1.5)
ax.text(p_hat, 4.55, f"MLE {p_hat:.1%}", ha="center", fontsize=9)
ax.set_yticks(range(len(variants)), [v[0] for v in variants], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("конверсия")
ax.set_xlim(0.06, 0.17)
ax.set_title("Одна задача: частотные CI и credible(плоский) практически совпадают,\n"
             "информативный априор Beta(8,92) тянет интервал к историческим 8%")
fig.suptitle("Шаг 1. Доверительный и credible-интервал на одной задаче",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_5_2_ci_vs_cri.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_5_2_ci_vs_cri.png")

# %% [markdown]
# Шаг 2. Base rate fallacy: тест с чувствительностью 95% и специфичностью 90%
# при превалентности 1%. Считаем «на 10 000 человек» (естественные частоты).

# %%
SENS, SPEC = 0.95, 0.90
PREV = 0.01
POP = 10_000
sick = POP * PREV
healthy = POP - sick
tp = sick * SENS
fn = sick - tp
fp = healthy * (1 - SPEC)
tn = healthy - fp
ppv = tp / (tp + fp)

print("=" * 78)
print("2) BASE RATE: медосмотр, чувствительность 95%, специфичность 90%")
print("-" * 78)
print(f"   На {POP:,} человек (превалентность {PREV:.0%}):")
print(f"     больны    {sick:>6,.0f}: тест + {tp:>6,.0f} | тест - {fn:>5,.0f}")
print(f"     здоровы {healthy:>7,.0f}: тест + {fp:>6,.0f} | тест - {tn:>5,.0f}")
print(f"   Позитивов всего {tp + fp:,.0f}, из них реально больны {tp:,.0f}:")
print(f"     P(болен | тест +) = {tp:,.0f}/{tp + fp:,.0f} = {ppv:.2%}")
print(f"   Интуиция говорила «95%» — на деле лишь каждый ~{round((tp + fp) / tp)}-й.")

# бизнес-прокси: антифрод «ЕдаДома»
prev_fraud, sens_fraud, fpr_fraud = 0.005, 0.90, 0.01
ppv_fraud = sens_fraud * prev_fraud / (sens_fraud * prev_fraud + fpr_fraud * (1 - prev_fraud))
print(f"\n   Антифрод «ЕдаДома»: фрод {prev_fraud:.1%}, чувствительность "
      f"{sens_fraud:.0%}, ложные срабатывания {fpr_fraud:.0%}")
print(f"     P(фрод | алерт) = {ppv_fraud:.1%} — {1 - ppv_fraud:.0%} алертов "
      f"замораживают честные заказы")
acc_trivial = 1 - prev_fraud
print(f"     «Точность 99.5%» ни о чём: модель «все заказы честные» даёт {acc_trivial:.1%}")

# %%
prevalences = np.logspace(-4, np.log10(0.5), 200)
ppvs = SENS * prevalences / (SENS * prevalences + (1 - SPEC) * (1 - prevalences))

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.3))
cats = ["больные,\nтест +", "больные,\nтест −", "здоровые,\nтест + (ложная тревога)",
        "здоровые,\nтест −"]
vals = [tp, fn, fp, tn]
colors = ["#C44E52", "#DD8452", "#937860", "#4C72B0"]
axes[0].bar(cats, vals, color=colors)
for v, x in zip(vals, range(4)):
    axes[0].text(x, v + 150, f"{v:,.0f}", ha="center", fontsize=9)
axes[0].set_ylabel("человек из 10 000")
axes[0].set_title(f"Естественные частоты: позитивов {tp + fp:,.0f}, из них больны лишь "
                  f"{tp:,.0f}\nP(болен|+) = {ppv:.2%} — тест «95% точности»")
axes[0].tick_params(axis="x", labelsize=8)

axes[1].semilogx(prevalences * 100, ppvs, lw=2.4, color="#4C72B0")
for pv, mk in [(0.01, "медосмотр: 1%"), (0.005, "фрод: 0.5%"), (0.10, "скрининг группы риска: 10%"),
               (0.50, "подтверждающий тест: 50%")]:
    ppv_v = SENS * pv / (SENS * pv + (1 - SPEC) * (1 - pv))
    axes[1].plot(pv * 100, ppv_v, "o", color="#C44E52", ms=7)
    axes[1].annotate(f"{mk}\nPPV {ppv_v:.0%}", (pv * 100, ppv_v),
                     textcoords="offset points", xytext=(8, -14), fontsize=8)
axes[1].axhline(0.95, color="gray", ls=":", lw=1.4)
axes[1].text(0.012, 0.955, "интуиция: «точность 95%»", fontsize=8, color="gray")
axes[1].set_xlabel("превалентность, % (лог. шкала)")
axes[1].set_ylabel("P(болен | тест +) = PPV")
axes[1].set_title("PPV падает вместе с превалентностью:\nредкое событие = большинство тревог ложные")
axes[1].set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_5_2_base_rate.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_5_2_base_rate.png")

# %% [markdown]
# Шаг 3. P(B > A) против p-value: A/B баннера, конверсия A = 600/5000 (12.0%),
# B = 645/5000 (12.9%). Плоские априоры Beta(1,1) -> два Beta-апостериора,
# сэмплим Монте-Карло и считаем прямые вероятностные высказывания.

# %%
K_A, N_AB, K_B = 600, 5000, 645
MC = 300_000
a_s = RNG.beta(K_A + 1, N_AB - K_A + 1, MC)
b_s = RNG.beta(K_B + 1, N_AB - K_B + 1, MC)
diff = b_s - a_s
p_b_beats_a = float(np.mean(diff > 0))
z, p_two, p_one = ztest_two_props(K_A, N_AB, K_B, N_AB)

print("=" * 78)
print("3) P(B>A) vs p-VALUE: A = 600/5000 (12.0%), B = 645/5000 (12.9%)")
print("-" * 78)
print(f"   частотный вывод : z = {z:.2f}, p (двустороннее) = {p_two:.3f} — не отвергаем H0")
print(f"   байесовский     : P(B>A) = {p_b_beats_a:.1%}  (Монте-Карло, {MC:,} сэмплов)")
print(f"                     ожидаемый лифт E[B-A] = {diff.mean():+.2%}")
print(f"                     P(B > A + 0.5 п.п.) = {np.mean(diff > 0.005):.1%}")
print(f"                     P(B < A - 1 п.п.)   = {np.mean(diff < -0.01):.1%}")
print(f"   соответствие    : P(B>A) = {p_b_beats_a:.3f} ~ 1 - p(одностороннее) = {1 - p_one:.3f}")

# словарь перевода: та же пара A, разные B
rows = []
for extra in (17, 33, 50, 66, 83, 99):
    k_b = K_A + extra
    z_i, p2_i, p1_i = ztest_two_props(K_A, N_AB, k_b, N_AB)
    b_i = RNG.beta(k_b + 1, N_AB - k_b + 1, MC)
    p_ba_i = float(np.mean(b_i > a_s))
    rows.append({
        "конверсия B": f"{k_b / N_AB:.2%}",
        "z": f"{z_i:.2f}",
        "p двустороннее": f"{p2_i:.3f}",
        "вердикт при α=5%": "не отвергаем" if p2_i > 0.05 else "отвергаем",
        "P(B>A)": f"{p_ba_i:.1%}",
        "1 − p одностор.": f"{1 - p1_i:.1%}",
        "_p_one_pct": p1_i * 100,
        "_pba_pct": p_ba_i * 100,
    })
df = pd.DataFrame(rows)
print("\n   СЛОВАРЬ ПЕРЕВОДА (одна и та же пара, разные данные):")
print(df.drop(columns=["_p_one_pct", "_pba_pct"]).to_string(index=False))

# чувствительность к скептическому априору «баннеры почти не работают»
print("\n   Чувствительность P(B>A) к априору (среднее 12%, разный вес):")
for w, (pa, pb) in {0: (1, 1), 1000: (120, 880), 5000: (600, 4400)}.items():
    a_w = RNG.beta(pa + K_A, pb + N_AB - K_A, MC)
    b_w = RNG.beta(pa + K_B, pb + N_AB - K_B, MC)
    print(f"     вес априора {w:>5} (в % выборки {w / N_AB:>5.0%}): "
          f"P(B>A) = {np.mean(b_w > a_w):.1%}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3))
axes[0].hist(diff, bins=120, color="#4C72B0", alpha=0.85)
axes[0].axvline(0, color="#C44E52", lw=2.5)
axes[0].text(0.0002, axes[0].get_ylim()[1] * 0.9,
             f"P(B<A) = {1 - p_b_beats_a:.1%}", color="#C44E52", fontsize=10)
axes[0].text(diff.mean(), axes[0].get_ylim()[1] * 0.55,
             f"P(B>A) = {p_b_beats_a:.1%}\nE[B−A] = {diff.mean():+.2%}",
             fontsize=10, ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd"))
axes[0].set_xlabel("B − A, разность конверсий")
axes[0].set_ylabel("плотность Монте-Карло")
axes[0].set_title(f"Апостериорное распределение эффекта: p-value = {p_two:.2f} «молчит»,\n"
                  f"а P(B>A) = {p_b_beats_a:.0%} — прямое высказывание о параметре")

p1_grid = np.linspace(0.001, 0.45, 45)
axes[1].plot(p1_grid * 100, 100 * (1 - p1_grid), "-", color="gray", lw=1.4,
             label="тождество 1 − p(одностороннее)")
axes[1].plot([r["_p_one_pct"] for r in rows], [r["_pba_pct"] for r in rows],
             "o", ms=8, color="#4C72B0", label="Монте-Карло P(B>A)")
for r in rows:
    axes[1].annotate(f"z={r['z']}", (r["_p_one_pct"], r["_pba_pct"]),
                     textcoords="offset points", xytext=(-14, 7), fontsize=8)
axes[1].set_xlabel("одностороннее p-value, % (= p двустороннее / 2)")
axes[1].set_ylabel("P(B>A), %")
axes[1].set_title("Словарь перевода: P(B>A) ≈ 1 − p(одностороннее)\n"
                  "при плоских априорах — «92%» звучит увереннее «p=0.17», но это одно и то же")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_5_2_p_b_over_a.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_5_2_p_b_over_a.png")

# %%
print()
print("=" * 78)
print("ГЛАВНОЕ (что должны увидеть):")
print("1) Credible при плоском априоре ~ Вильсон (расхождение доли процента):")
print("   числа одинаковые, интерпретации разные. Информативный априор — уже")
print("   другое число: интервал уезжает к исторической конверсии.")
print("2) Base rate: тест с чувствительностью 95% при превалентности 1% даёт")
print("   P(болен|+) = 8.76%: из ~11 позитивов болен один. Без превалентности")
print("   «точность» не интерпретируется — и в медицине, и во фрод-скоринге.")
print("3) P(B>A) ~ 1 - p(одностороннее): при плоских априорах это перевод,")
print("   а не новая информация. Новое — остальные вопросы: P(B>A+0.5 п.п.),")
print("   ожидаемый лифт, риск просадки. Скептический априор двигает P(B>A)")
print("   к 50%, но даже с весом целой выборки не отменяет данные (Кромвель).")
