# -*- coding: utf-8 -*-
"""Практика 4.4 — Интерпретация и решение: полный пайплайн (кейс «ЕдаДома»).

Финальная сборка модуля: от сырых логов до вердикта roll/hold/iterate.

1) логи заказов и событий -> юнит-таблица (метрика на юзере);
2) валидационные чеки (SRM, инвариант) -> только потом тесты;
3) тесты: OEC (ARPU, t-тест + бутстреп-CI), 2 guardrail, 3 secondary
   (ratio через дельта-метод) с поправкой Холма;
4) локализация эффекта: децили по пре-периодной выручке — где живёт эффект;
5) сегменты «до» и «после» поправки Холма: ложный сегмент-победитель;
6) report() — шаблон карточки для стейкхолдеров + решение roll/hold/iterate.

Запуск:  python3 practice_4_4.py
Графики (PNG) сохраняются рядом со скриптом.
"""

# %% [markdown]
# # Практика 4.4 — От сырых логов до вердикта
# Тест «умные рекомендации в корзине» (сквозной кейс 4.1): в дизайн-доке
# договорились — OEC = ARPU, guardrails = доля опоздавших заказов и обращения
# в поддержку, secondary = CTR рекомендаций / конверсия в заказ / средний чек.
# Порог практической значимости: +3% ARPU (окупает поддержку рекомендера).

# %%
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(44)  # seed -> результат воспроизводим
HERE = Path(__file__).resolve().parent
ALPHA = 0.05
PRACTICAL_MIN = 0.03  # порог практической значимости: +3% ARPU


def prop_z(k1, n1, k2, n2):
    """Двухвыборочный z-тест долей."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return p1, p2, z, 2 * stats.norm.sf(abs(z))


def holm(pvals):
    """Поправка Холма (контроль FWER): скорректированные p и флаги отвержения."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m, run, adj = len(p), 0.0, np.empty(len(p))
    for r, idx in enumerate(order):
        run = max(run, (m - r) * p[idx])
        adj[idx] = min(1.0, run)
    return adj, adj < ALPHA


def ratio_delta(num_t, den_t, num_c, den_c):
    """Дельта-метод для ratio-метрик (3.3): R = sum(num)/sum(den) по юзерам."""
    R = {}
    for key, num, den in (("t", num_t, den_t), ("c", num_c, den_c)):
        r = num.sum() / den.sum()
        v = num - r * den  # влияющие функции юзеров
        R[key] = (r, np.sqrt(v.var() / len(num)) / den.mean())
    z = (R["t"][0] - R["c"][0]) / np.hypot(R["t"][1], R["c"][1])
    return R["c"][0], R["t"][0], z, 2 * stats.norm.sf(abs(z))


# %% [markdown]
# ## Шаг 1. Сырые логи -> юнит-таблица
# Генерируем 2 недели логов: заказы (выручка, минуты доставки), события
# рекомендаций, пре-периодная выручка юзера (для децилей). Эффект заложен
# в ЧИСЛО заказов и растёт по децилям пре-выручки: D1–D6 +1%, D7–D8 +4%,
# D9 +8%, D10 +14% — рекомендации сильнее всего помогают «китам».

# %%
N_PER = 40_000
n = 2 * N_PER

user_id = np.arange(n)
variant = RNG.integers(0, 2, n)                       # 0 = контроль, 1 = тест
platform = RNG.choice(["ios", "android", "web"], n, p=[0.45, 0.35, 0.20])
is_new = RNG.choice([1, 0], n, p=[0.25, 0.75])
geo = RNG.choice(["msk", "region"], n, p=[0.40, 0.60])

u = RNG.normal(0, 0.40, n)                            # латентная тяга юзера
lam_base = 1.6 * np.exp(u)                            # среднее число заказов/2 нед


def draw_checks(k):
    """k значений чека: логнормаль (медиана 900, тянет среднее к ~1046 ₽)."""
    return RNG.lognormal(np.log(900), 0.55, k)


# пре-период: та же механика, эффекта нет
pre_orders = RNG.poisson(lam_base)
pre_rev = np.bincount(np.repeat(user_id, pre_orders), weights=draw_checks(pre_orders.sum()), minlength=n)
decile = np.minimum((pd.Series(pre_rev).rank(method="first", pct=True) * 10).astype(int), 9)  # 0..9

mult = np.where(decile <= 5, 1.01, np.where(decile <= 7, 1.04, np.where(decile == 8, 1.08, 1.14)))
orders = RNG.poisson(lam_base * np.where(variant == 1, mult, 1.0))

# сырые логи заказов (то, что лежит в хранилище)
tot = orders.sum()
own = np.repeat(user_id, orders)                      # user_id каждого заказа
order_rev = draw_checks(tot)
delivery = RNG.normal(34.0, 8, tot) + 0.25 * variant[own]
late = (delivery > 45).astype(float)
events = pd.DataFrame({
    "order_id": np.arange(tot), "user_id": own, "day": RNG.integers(0, 14, tot),
    "revenue": order_rev.round(0), "delivery_min": delivery.round(1), "late": late,
})

# юнит-таблица: метрики на юзере (join логов с атрибутами юзера)
revenue = np.bincount(own, weights=order_rev, minlength=n)
late_sum = np.bincount(own, weights=late, minlength=n)
exposures = RNG.poisson(8, n)
clicks = RNG.binomial(exposures, np.where(variant == 1, 0.108, 0.100))
support = RNG.binomial(1, np.where(variant == 1, 0.052, 0.050))

users = pd.DataFrame({
    "user_id": user_id, "variant": variant, "platform": platform, "is_new": is_new,
    "geo": geo, "pre_rev": pre_rev, "decile": decile, "exposures": exposures,
    "clicks": clicks, "support": support, "orders": orders, "revenue": revenue,
    "late_share": np.divide(late_sum, orders, out=np.zeros(n), where=orders > 0),
})

print("=" * 96)
print(f"1) Логи: {tot:,} заказов у {n:,} юзеров · сплит 50/50 · 2 недели".replace(",", " "))
print(events.head(5).to_string(index=False))
print("--- юнит-таблица (агрегат логов на юзера):")
print(users[["user_id", "variant", "platform", "pre_rev", "orders", "revenue", "late_share"]].head(3).to_string(index=False))

# %% [markdown]
# ## Шаг 2. Валидация: SRM и инварианты — до всяких тестов
# Порядок урока 4.3: чек-лист валидности зелёный -> только потом p-value.

# %%
n_c, n_t = (variant == 0).sum(), (variant == 1).sum()
chi2_srm, p_srm = stats.chisquare([n_t, n_c], [n / 2, n / 2])[:2]
t_inv, p_inv = stats.ttest_ind(exposures[variant == 1], exposures[variant == 0], equal_var=False)

print("=" * 96)
print("2) Валидация")
print(f"   SRM: {n_t:,} / {n_c:,} — chi2 p = {p_srm:.2f} (порог тревоги 0.001) — ok".replace(",", " "))
print(f"   Инвариант «показы рекомендаций на юзера»: {exposures[variant == 1].mean():.2f} vs "
      f"{exposures[variant == 0].mean():.2f}, p = {p_inv:.2f} — ok")
print("   Телеметрия/интерференция/длительность — зелёные по чек-листу 4.3. Читаем результаты.")

# %% [markdown]
# ## Шаг 3. Метрики и тесты: OEC + guardrails + secondary (Холм)
# OEC — один, без поправок; guardrails — смотрим на просадку; вторичные —
# семья из 3 гипотез, обязателен Холм (2.4). Ratio-метрики (CTR, чек) —
# дельта-метод (3.3). CI для OEC — перцентильный бутстреп по юзерам (1.4).

# %%
c, t = users[users.variant == 0], users[users.variant == 1]

tt = stats.ttest_ind(t.revenue, c.revenue, equal_var=False)
oec_diff = t.revenue.mean() - c.revenue.mean()
boot = np.array([RNG.choice(t.revenue, N_PER, replace=True).mean()
                 - RNG.choice(c.revenue, N_PER, replace=True).mean() for _ in range(1_000)])
ci = np.percentile(boot, [2.5, 97.5])
oec_rel = oec_diff / c.revenue.mean()

g1 = stats.ttest_ind(t.late_share, c.late_share, equal_var=False)
_, _, z_g2, p_g2 = prop_z(t.support.sum(), N_PER, c.support.sum(), N_PER)

ctr_c, ctr_t, z_ctr, p_ctr = ratio_delta(t.clicks, t.exposures, c.clicks, c.exposures)
_, _, z_cr, p_cr = prop_z((t.orders > 0).sum(), N_PER, (c.orders > 0).sum(), N_PER)
chk_c, chk_t, z_chk, p_chk = ratio_delta(t.revenue, t.orders, c.revenue, c.orders)

sec_adj, sec_rej = holm(np.array([p_ctr, p_cr, p_chk]))

tab3 = pd.DataFrame({
    "роль": ["OEC", "guardrail", "guardrail", "secondary", "secondary", "secondary"],
    "контроль": [c.revenue.mean(), c.late_share.mean(), c.support.mean(),
                 ctr_c, (c.orders > 0).mean(), chk_c],
    "тест": [t.revenue.mean(), t.late_share.mean(), t.support.mean(),
             ctr_t, (t.orders > 0).mean(), chk_t],
    "эффект": [f"{oec_rel:+.1%}",
               f"{(t.late_share.mean() - c.late_share.mean()) * 100:+.2f} п.п.",
               f"{(t.support.mean() - c.support.mean()) * 100:+.2f} п.п.",
               f"{(ctr_t / ctr_c - 1):+.1%}",
               f"{((t.orders > 0).mean() - (c.orders > 0).mean()) * 100:+.2f} п.п.",
               f"{(chk_t / chk_c - 1):+.1%}"],
    "p": [tt.pvalue, g1.pvalue, p_g2, p_ctr, p_cr, p_chk],
    "p (Холм)": [np.nan, np.nan, np.nan, sec_adj[0], sec_adj[1], sec_adj[2]],
}, index=["ARPU, ₽/юзер", "доля опозд. заказов", "обращения в поддержку",
          "CTR рекомендаций", "конверсия в заказ", "средний чек, ₽"])

print("=" * 96)
print("3) Результаты (secondary — с поправкой Холма на семью из 3 гипотез)")
print(tab3.round(4).to_string())
print(f"   OEC: {oec_diff:+.0f} ₽ ({oec_rel:+.1%}), 95% CI (бутстреп): [{ci[0]:+.0f}, {ci[1]:+.0f}] ₽,")
print(f"   нижняя граница CI = {ci[0] / c.revenue.mean():+.1%} ARPU против порога практической значимости {PRACTICAL_MIN:.0%}")

# %% [markdown]
# ## Шаг 4. Локализация эффекта: децили по пре-выручке
# Где эффект живёт — в теле или в хвосте? Децили — строго по ПРЕ-периоду
# (нарезка по post-метрике = selection, 4.3). Эффект растёт к старшим
# децилям: винзоризация по P99 урезала бы не шум, а сам эффект (3.4).

# %%
rows = []
for d in range(10):
    cd, td = c[c.decile == d], t[t.decile == d]
    diff = td.revenue.mean() - cd.revenue.mean()
    se = np.sqrt(td.revenue.var() / len(td) + cd.revenue.var() / len(cd))
    rows.append({"дециль": f"D{d + 1}", "n/вариант": len(cd),
                 "ARPU ctrl, ₽": cd.revenue.mean(),
                 "эффект, %": diff / cd.revenue.mean() * 100,
                 "CI низ, %": (diff - 1.96 * se) / cd.revenue.mean() * 100,
                 "CI верх, %": (diff + 1.96 * se) / cd.revenue.mean() * 100})
tab4 = pd.DataFrame(rows).set_index("дециль")

print("=" * 96)
print("4) Локализация: эффект ARPU по децилям пре-периодной выручки")
print(tab4.round(1).to_string())
print("   Эффект концентрируется в хвосте (D9–D10): там живёт большая часть прироста.")
print("   Проверка 3.4: агрессивная винзоризация (P95 и жёстче) срезала бы именно его.")

# %%
fig, ax = plt.subplots(figsize=(9.4, 4.3))
eff = tab4["эффект, %"].values
err = np.vstack([eff - tab4["CI низ, %"].values, tab4["CI верх, %"].values - eff])
ax.bar(range(10), eff, 0.62, color="#4C72B0")
ax.errorbar(range(10), eff, yerr=err, fmt="none", ecolor="#C44E52", elinewidth=1.6, capsize=3)
ax.axhline(oec_rel * 100, color="black", ls="--", lw=1.6, label=f"средний эффект {oec_rel:+.1%}")
ax.set_xticks(range(10), [f"D{i + 1}" for i in range(10)])
ax.set_xlabel("дециль выручки пре-периода (D10 — «киты»)")
ax.set_ylabel("эффект ARPU, %")
ax.set_title("Локализация эффекта: прирост живёт в старших децилях —\nвинзоризация хвоста резала бы именно его")
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "practice_4_4_deciles.png", dpi=150)
print("Сохранено: practice_4_4_deciles.png")

# %% [markdown]
# ## Шаг 5. Сегменты «до» и «после» поправки Холма
# а) 12 заявленных в дизайн-доке сегментов (платформа x новизна x гео):
#    эффект в них наследуется от OEC, но сегменты малы — сырой максимум
#    завышен (winner's curse), после Холма часть «открытий» умирает.
# б) 20 псевдосегментов в A/A-режиме (хэш id внутри контроля): истинный
#    эффект НОЛЬ по построению — смотрим, найдёт ли «сегмент-победитель»
#    сырой анализ и убьёт ли его Холм.

# %%
seg_rows = []
for (plat, nw, g), grp in users.groupby(["platform", "is_new", "geo"]):
    gc, gt = grp[grp.variant == 0].revenue, grp[grp.variant == 1].revenue
    seg_rows.append({"сегмент": f"{plat}|{'нов' if nw else 'стар'}|{g}", "n/вариант": len(gc),
                     "эффект, %": (gt.mean() / gc.mean() - 1) * 100,
                     "p сырой": stats.ttest_ind(gt, gc, equal_var=False).pvalue})
tab5a = pd.DataFrame(seg_rows).set_index("сегмент")
adj5a, rej5a = holm(tab5a["p сырой"].values)
tab5a["p Холм"] = adj5a
tab5a["значим"] = np.where(rej5a, "да", "нет")
best = tab5a["эффект, %"].idxmax()

# б) A/A внутри контроля: pseudo-сплит + 20 псевдосегментов
rng_aa = np.random.default_rng(2024)
ctrl = users[users.variant == 0].copy()
ctrl["pseudo"] = rng_aa.integers(0, 2, len(ctrl))
ctrl["pseg"] = rng_aa.integers(0, 20, len(ctrl))
aa_rows = []
for s, grp in ctrl.groupby("pseg"):
    gc, gt = grp[grp.pseudo == 0].revenue, grp[grp.pseudo == 1].revenue
    aa_rows.append({"pseg": s, "эффект, %": (gt.mean() / gc.mean() - 1) * 100,
                    "p сырой": stats.ttest_ind(gt, gc, equal_var=False).pvalue})
tab5b = pd.DataFrame(aa_rows).set_index("pseg")
tab5b["p Холм"] = holm(tab5b["p сырой"].values)[0]

print("=" * 96)
print("5а) 12 реальных сегментов (заявлены до теста; эффект наследуется от OEC)")
print(tab5a.sort_values("эффект, %", ascending=False).round(4).to_string())
print(f"   Сырой «победитель»: {best}: {tab5a.loc[best, 'эффект, %']:+.1f}% против {oec_rel:+.1%} на OEC —")
print("   завышение = winner's curse: максимум по 12 шумным оценкам всегда оптимистичен.")
print()
print("5б) 20 псевдосегментов в A/A (истинный эффект = 0 по построению), топ-5 по p:")
print(tab5b.sort_values("p сырой").head(5).round(4).to_string())
n_raw = (tab5b["p сырой"] < ALPHA).sum()
n_adj = (tab5b["p Холм"] < ALPHA).sum()
if n_raw:
    s0 = tab5b["p сырой"].idxmin()
    print(f"   Сырой «победитель»: псевдосегмент {s0}: {tab5b.loc[s0, 'эффект, %']:+.1f}%, "
          f"p = {tab5b.loc[s0, 'p сырой']:.3f} -> после Холма p = {tab5b.loc[s0, 'p Холм']:.2f}")
print(f"   Значимых: сырой анализ {n_raw} из 20, после Холма {n_adj} — ложный победитель исчезает.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
srt = tab5a.sort_values("эффект, %")
axes[0].barh(range(len(srt)), srt["эффект, %"], color="#4C72B0")
axes[0].axvline(oec_rel * 100, color="black", ls="--", lw=1.6, label=f"OEC {oec_rel:+.1%}")
axes[0].set_yticks(range(len(srt)), srt.index, fontsize=8)
axes[0].set_xlabel("эффект ARPU, %")
axes[0].set_title("Реальные сегменты: сырой максимум завышен\n(winner's curse) — сравнивать с OEC")
axes[0].legend(fontsize=9)
order = np.argsort(tab5b["p сырой"].values)
axes[1].plot(range(1, 21), tab5b["p сырой"].values[order], "o-", color="#4C72B0", label="p сырой")
axes[1].plot(range(1, 21), tab5b["p Холм"].values[order], "s--", color="#C44E52", label="p после Холма")
axes[1].axhline(ALPHA, color="black", ls=":", lw=1.6, label="alpha = 0.05")
axes[1].set_xlabel("псевдосегменты A/A (отсортированы по p)")
axes[1].set_ylabel("p-value")
axes[1].set_title(f"A/A: 20 сегментов под H0 — сырых «значимых» {n_raw},\nпосле Холма {n_adj}")
axes[1].legend(fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "practice_4_4_segments.png", dpi=150)
print("Сохранено: practice_4_4_segments.png")

# %% [markdown]
# ## Шаг 6. report() — карточка для стейкхолдеров и решение
# Шаблон: эффект + CI + guardrails + локализация + деньги + рекомендация.
# Решение — по правилу из дизайн-дока (4.1), а не «по прокрасу».

# %%
def decide(p_oec, ci_low_rel, practical, guardrails_ok):
    """Decision rule 2x2 из 4.1 + порог практической значимости из 3.1."""
    if not guardrails_ok and p_oec < ALPHA:
        return ("ITERATE — OEC значим, но guardrail просел: раскатка недопустима; "
                "чинить причину просадки и перетестировать")
    if not guardrails_ok:
        return "HOLD — guardrail просел при отсутствии эффекта, раскатка недопустима"
    if p_oec < ALPHA and ci_low_rel >= practical:
        return "ROLL — эффект значим, нижняя граница CI выше порога практической значимости"
    if p_oec < ALPHA:
        return "ROLL с мониторингом — значим, но нижняя CI ниже порога: 2 недели следить за guardrails"
    return "ITERATE — значимости нет: смотреть дизайн, чувствительность (CUPED/3.5), не продлевать вслепую"


def report(name, hyp, oec, guardrails, secondary, localization, money, decision):
    """Карточка теста для стейкхолдеров: одно действие — одно число."""
    line = "=" * 96
    out = [line, f"ОТЧЁТ: {name}", f"Гипотеза: {hyp}", line]
    (oec_name, ctrl, tst, diff, ci_, p, rel) = oec
    out.append(f"OEC {oec_name}: {ctrl:,.0f} -> {tst:,.0f} ₽/юзер  ({diff:+,.0f} ₽, {rel:+.1%})".replace(",", " "))
    out.append(f"   95% CI: [{ci_[0]:+,.0f}, {ci_[1]:+,.0f}] ₽ ({ci_[0] / ctrl:+.1%}..{ci_[1] / ctrl:+.1%}),  p = {p:.1e}".replace(",", " "))
    for gname, gc, gt, gp, ok in guardrails:
        gp_txt = f"p = {gp:.2f}" if gp >= 0.001 else "p < 0.001"
        out.append(f"Guardrail {gname}: {gc:.4f} -> {gt:.4f}  ({gp_txt})  {'OK, не просел' if ok else 'ПРОСАДКА'}")
    for sname, sc, st_, sig in secondary:
        out.append(f"Secondary {sname}: {sc:.4f} -> {st_:.4f}  [{'значим (Холм)' if sig else 'без изменений'}]")
    out.append(f"Локализация: {localization}")
    out.append(f"Экономика: {money}")
    out.append(f"РЕШЕНИЕ: {decision}")
    out.append(line)
    return "\n".join(out)


users_month, windows = 600_000, 2  # активная база и 2-недельных окон в месяце
money = (f"+{oec_diff:,.0f} ₽/юзер за окно x {windows} окна x {users_month:,} юзеров "
         f"= +{oec_diff * windows * users_month / 1e6:.0f} млн ₽/мес — упущенный доход пока guardrail красный"
         ).replace(",", " ")
guard = [
    ("доля опоздавших заказов", c.late_share.mean(), t.late_share.mean(), g1.pvalue, g1.pvalue > ALPHA),
    ("обращения в поддержку", c.support.mean(), t.support.mean(), p_g2, p_g2 > ALPHA),
]
second = [("CTR рекомендаций", ctr_c, ctr_t, sec_rej[0]),
          ("конверсия в заказ", (c.orders > 0).mean(), (t.orders > 0).mean(), sec_rej[1]),
          ("средний чек", chk_c, chk_t, sec_rej[2])]
loc_txt = (f"эффект растёт по децилям пре-выручки (D1 {tab4.loc['D1', 'эффект, %']:+.0f}%, "
           f"D9 {tab4.loc['D9', 'эффект, %']:+.0f}%, D10 {tab4.loc['D10', 'эффект, %']:+.0f}%) — живёт в хвосте")
verdict = decide(tt.pvalue, ci[0] / c.revenue.mean(), PRACTICAL_MIN, g1.pvalue > ALPHA and p_g2 > ALPHA)

print(report(
    "«Умные рекомендации в корзине» — 2 недели, 40k/40k, alpha=5%, MDE 3%",
    "персональные рекомендации в корзине (X) увеличивают частоту заказов (Y) и ARPU (Z)",
    ("ARPU", c.revenue.mean(), t.revenue.mean(), oec_diff, ci, tt.pvalue, oec_rel),
    guard, second, loc_txt, money, verdict))

# %%
print()
print("=" * 96)
print("ГЛАВНОЕ (что должны увидеть):")
print(f"1) OEC ARPU: {oec_diff:+.0f} ₽ ({oec_rel:+.1%}), CI [{ci[0]:+.0f}, {ci[1]:+.0f}] ₽ — не «прокрас», а величина с диапазоном.")
print(f"2) Guardrail «доля опозданий» просел ({c.late_share.mean():.3f} -> {t.late_share.mean():.3f}, p < 0.001):")
print("   спрос вырос -> нагрузка на курьеров -> доставка дольше. Decision rule 2x2 (4.1): OEC вверх +")
print("   guardrail вниз = НЕ раскатывать. Деньги не отменяют guardrail — это и есть «по-взрослому».")
print(f"3) Secondary после Холма: CTR {(ctr_t / ctr_c - 1):+.1%} значим, конверсия "
      f"{((t.orders > 0).mean() - (c.orders > 0).mean()) * 100:+.2f} п.п. и чек {(chk_t / chk_c - 1):+.1%} — нет:")
print("   направление сонаправлено (клики -> конверсия -> выручка), чувствительность выше у механики (айсберг).")
print(f"4) Локализация: эффект в D9–D10 ({tab4.loc['D10', 'эффект, %']:+.0f}%) — резать хвост нельзя (3.4).")
print(f"5) Сегменты: сырой победитель {tab5a['эффект, %'].max():+.1f}% vs OEC {oec_rel:+.1%} (winner's curse);")
print(f"   в A/A из 20 псевдосегментов сырых «значимых» {n_raw}, после Холма — {n_adj}.")
print(f"6) Вердикт: {verdict.split(' — ')[0]} — эффект {oec_diff * windows * users_month / 1e6:.0f} млн ₽/мес есть, "
      f"но сначала чиним доставку.")
