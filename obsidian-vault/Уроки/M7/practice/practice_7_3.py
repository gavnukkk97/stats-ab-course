# -*- coding: utf-8 -*-
"""Практика 7.3 — Интерфейс платформы: от гипотезы до вердикта
(сквозной кейс «ЕдаДома», урок 7.3 модуля «Проектирование АБ-платформы»).

Принцип курса: не верь интуиции — проверь симуляцией.

Что делаем: мини-платформа экспериментов в одном скрипте.
  class ExperimentPlatform:
    submit(design_doc)  -> валидация заявки: метрики из метрик-хаба,
                           автоподсчёт n из MDE (формула 3.1), guardrail-пакет,
                           проверка занятости слоёв (пересечения);
    run(ticket, world)  -> симуляция теста по дням с гейтами:
                           SRM-гейт (хи-квадрат, порог 0.001 — урок 4.2),
                           инварианты (телеметрия), авто-стоп по guardrail;
    analyze(run)        -> OEC: Уэлч + CUPED (урок 3.5); вторичные + Холм (2.4);
    verdict(analysis)   -> decision rule 2x2 из урока 4.1;
    report(...)         -> строка отчёта в «ленту результатов».
  Прогоняем 3 сценария: чистый зелёный / авто-стоп по guardrail / серый.
  Бонус: SRM-гейт отклоняет сломанный тест до всякого анализа.

Запуск из корня репозитория: python3 course/modules/M7/practice/practice_7_3.py
График (PNG) сохраняется в images/ модуля.
"""

# %% [markdown]
# Шаг 0. Импорты и константы «ЕдаДома»: α=5%, мощность 80%, трафик
# 11 700 юзеров/неделю на обе группы (урок 3.1).

# %%
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.special import expit

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

ALPHA, POWER = 0.05, 0.80
Z_A = stats.norm.ppf(1 - ALPHA / 2)   # 1.96
Z_B = stats.norm.ppf(POWER)           # 0.84
WEEKLY = 11_700                      # юзеров в неделю на обе группы
DAILY = WEEKLY // 7                  # ~1671/день
SRM_ALPHA = 0.001                    # порог SRM-гейта (урок 4.2)
GR_Z = None  # граница задаётся числом плановых просмотров каждого теста
U_COEF = 1.3                         # сила «латентной активности» юзера
U_CAL = stats.norm.ppf(np.arange(1, 20_001) / 20_001)  # детерминированные N(0,1)


def seed_of(name: str) -> int:
    """Детерминированный seed из строки (hash() в Python солёный — нельзя)."""
    return zlib.crc32(name.encode("utf-8"))


def n_per_group(mde, sigma, alpha=ALPHA, power=POWER):
    """n на группу: двухвыборочная формула урока 3.1 (с множителем 2)."""
    z1 = stats.norm.ppf(1 - alpha / 2)
    z2 = stats.norm.ppf(power)
    return 2 * (z1 + z2) ** 2 * sigma ** 2 / mde ** 2


# %% [markdown]
# Шаг 1. Класс платформы. Заявка = дизайн-док из урока 4.1 (гипотеза X->Y->Z,
# OEC, MDE, guardrails, слой). Платформа — не «сервис тестов», а конвейер
# с валидацией на входе, гейтами по ходу и вердиктом по зафиксированному
# правилу на выходе (Microsoft ExP: Portal / Execution / Analysis / Reporting).

# %%
@dataclass
class DesignDoc:
    """Заявка = дизайн-док (урок 4.1): минимум полей, всё решающее — до запуска."""
    exp_id: str
    name: str
    layer: str                     # слой = домен изменений (урок 4.2)
    oec: str                       # метрика решения (из метрик-хаба)
    mde_rel: float                 # MDE, % отн. — из экономики (уроки 3.1, 4.5)
    hyp_x: str = ""
    hyp_y: str = ""
    hyp_z: str = ""
    guardrails: tuple = ()         # если пусто — автоподстановка пакета платформы
    secondary: tuple = ()          # вторичные метрики (Холм на анализе)
    share: float = 1.0             # доля трафика слоя под тест (A+B вместе)


@dataclass
class RunResult:
    exp_id: str
    planned_days: int
    days: list = field(default_factory=list)
    n_c: list = field(default_factory=list)
    n_t: list = field(default_factory=list)
    p_srm: list = field(default_factory=list)
    z_inv: list = field(default_factory=list)
    z_gr: list = field(default_factory=list)
    gr_diff: list = field(default_factory=list)
    oec_path: list = field(default_factory=list)
    stop_day: int = 0
    stop_reason: str = ""
    stopped_early: bool = False
    data: dict = field(default_factory=dict)
    p_srm_last: float = 1.0

    def track(self, day, nc, nt, p_srm, z_inv, z_gr, gr_diff, oec_cum):
        self.days.append(day); self.n_c.append(nc); self.n_t.append(nt)
        self.p_srm.append(p_srm); self.z_inv.append(z_inv)
        self.z_gr.append(z_gr); self.gr_diff.append(gr_diff)
        self.oec_path.append(oec_cum)
        self.p_srm_last = p_srm

    def stop(self, day, reason):
        self.stop_day = day
        self.stop_reason = reason
        self.stopped_early = day < self.planned_days


class ExperimentPlatform:
    """Мини-платформа: submit -> run -> analyze -> verdict -> report.

    Собирает пройденное: MDE/n (3.1), CUPED (3.5), Холм (2.4), SRM (4.2),
    decision rule 2x2 (4.1). Новое — интерфейс: валидация заявки,
    гейты по ходу, лента результатов.
    """

    # метрик-хаб: единые определения метрик с паспортом (Airbnb Certified)
    METRIC_HUB = {
        "конверсия в заказ (2 нед.)": dict(p0=0.230, kind="binary"),
        "открытие пуша (7 дн. окно)": dict(p0=0.062, kind="binary"),
        "выручка/юзер (винзор. P99)": dict(p0=201.0, sd=580.0, kind="cont"),
        "CTR карточки (нед.)": dict(p0=0.576, kind="binary"),
        "доля отмен заказов": dict(p0=0.048, kind="binary"),
        "доля юзеров с ошибкой (crash/JS)": dict(p0=0.020, kind="binary"),
    }
    GUARDRAIL_PACK = ("доля юзеров с ошибкой (crash/JS)",)
    LAYERS = {"ранжирование": [], "интерфейс": [], "коммуникации": []}  # занятость

    def __init__(self):
        self.LAYERS = {name: [] for name in type(self).LAYERS}

    def release(self, exp_id):
        for name in self.LAYERS:
            self.LAYERS[name] = [(key, share) for key, share in self.LAYERS[name] if key != exp_id]

    # ---------------- submit: валидация заявки ----------------
    def submit(self, doc: DesignDoc):
        """Валидация: метрики существуют, MDE -> n -> длительность,
        guardrails назначены, слой не переполнен. Возвращает тикет или отказ."""
        problems = []
        if doc.layer not in self.LAYERS or not 0 < doc.share <= 1 or doc.mde_rel <= 0:
            raise ValueError("Некорректный слой, доля трафика или MDE")
        if any(doc.exp_id == key for slots in self.LAYERS.values() for key, _ in slots):
            raise ValueError("Заявка уже резервирует слот")
        if doc.oec not in self.METRIC_HUB:
            problems.append(f"OEC «{doc.oec}» отсутствует в метрик-хабе")
            print(f"[{doc.exp_id}] ОТКАЗ: {problems[0]}")
            return None
        m = self.METRIC_HUB[doc.oec]
        if m["kind"] == "binary":
            sigma = math.sqrt(m["p0"] * (1 - m["p0"]))
        else:
            sigma = m["sd"]
        mde_abs = doc.mde_rel * m["p0"]
        n = math.ceil(n_per_group(mde_abs, sigma) * 1.05)  # запас 5% (урок 3.1)
        weeks = max(2, math.ceil(2 * n / (WEEKLY * doc.share)))

        guardrails = doc.guardrails or self.GUARDRAIL_PACK  # автоподстановка
        for g in guardrails:
            if g not in self.GUARDRAIL_PACK:
                problems.append(f"guardrail «{g}» не реализован в этой учебной симуляции")

        # пересечения в слое: суммарная занятость + share <= 100% (урок 4.2)
        busy = sum(s for (_, s) in self.LAYERS[doc.layer])
        if busy + doc.share > 1.0 + 1e-9:
            free = 1.0 - busy
            print(f"[{doc.exp_id}] ОТКАЗ: слой «{doc.layer}» переполнен — занято "
                  f"{busy:.0%}, свободно {free:.0%} < доля теста {doc.share:.0%}; "
                  f"ставим в очередь (урок 7.4)")
            return None
        if problems:
            print(f"[{doc.exp_id}] ОТКАЗ: " + "; ".join(problems))
            return None

        self.LAYERS[doc.layer].append((doc.exp_id, doc.share))
        ticket = dict(doc=doc, n=n, weeks=weeks, guardrails=guardrails,
                      mde_abs=mde_abs, p0=m["p0"], kind=m["kind"])
        print(f"[{doc.exp_id}] ПРИНЯТ: «{doc.name}» | слой={doc.layer} "
              f"(занято {busy + doc.share:.0%}) | OEC={doc.oec} (p0={m['p0']:g}) "
              f"| MDE=+{doc.mde_rel:.1%} отн. -> n={n:,}/группу (автоподсчёт) "
              f"-> {weeks} нед. | guardrails: {', '.join(guardrails)}")
        return ticket

    # ---------------- run: симуляция теста с гейтами ----------------
    def run(self, ticket, effect_rel=0.0, err_mult=1.0, drop_test_share=0.0, *, rng_seed=None):
        """Симулирует тест по дням. Гейты каждый день:
        1) SRM — хи-квадрат долей против плана 50/50, порог 0.001/K для K плановых дней;
        2) инвариант «событий в пре-периоде» — до воздействия;
        3) guardrail «доля юзеров с ошибкой» — авто-стоп, если рост значимо
           выше порога +0.3 п.п. (z > norm.isf(alpha/K); K задано до старта).
        Истинный эффект OEC задаётся в % отн. (effect_rel).
        """
        doc = ticket["doc"]
        days = 7 * ticket["weeks"]
        # интерсепт подбираем так, чтобы E[expit(a0 + U*u)] = p0 (Jensen!)
        a0 = brentq(lambda a: expit(a + U_COEF * U_CAL).mean() - ticket["p0"],
                    -12.0, 6.0)
        add_total = max(1, int(DAILY * doc.share))
        run = RunResult(exp_id=doc.exp_id, planned_days=days)
        # Заранее ограниченное число просмотров. Bonferroni не требует их независимости.
        srm_cutoff = SRM_ALPHA / days
        invariant_z = stats.norm.isf(SRM_ALPHA / (2*days))
        guardrail_z = stats.norm.isf(ALPHA / days)
        rng = np.random.default_rng(seed_of(doc.exp_id) % 2 ** 31 if rng_seed is None else rng_seed)

        y_c, y_t, pre_c, pre_t, er_c, er_t, ev_c, ev_t = ([] for _ in range(8))
        for day in range(1, days + 1):
            u = rng.normal(0, 1, add_total)
            p_u = expit(a0 + U_COEF * u)                     # «латентная» база юзера
            pre = rng.binomial(1, p_u)                       # пре-период (CUPED)
            y_ctrl = rng.binomial(1, p_u)
            y_test = rng.binomial(1, np.clip(p_u * (1 + effect_rel), 0, 1))
            err_ctrl = rng.binomial(1, 0.020, add_total)     # guardrail: ошибки
            err_test = rng.binomial(1, 0.020 * err_mult, add_total)
            ev = rng.poisson(2.3, add_total)                 # инвариант: число событий до назначения
            keep = rng.random(add_total) >= drop_test_share  # «теряем» тест-группу

            assigned = rng.binomial(1, 0.5, add_total).astype(bool)
            control_mask, treatment_mask = ~assigned, assigned & keep
            y_c.append(y_ctrl[control_mask]); y_t.append(y_test[treatment_mask])
            pre_c.append(pre[control_mask]); pre_t.append(pre[treatment_mask])
            er_c.append(err_ctrl[control_mask]); er_t.append(err_test[treatment_mask])
            ev_c.append(ev[control_mask]); ev_t.append(ev[treatment_mask])

            nc, nt = sum(map(len, y_c)), sum(map(len, y_t))
            # Гейт 1: SRM (хи-квадрат, урок 4.2)
            chi2 = (nc - nt) ** 2 / (nc + nt)
            p_srm = stats.chi2.sf(chi2, 1)
            # Гейт 2: инвариант — события на юзера не должны различаться
            e_c = np.concatenate(ev_c); e_t = np.concatenate(ev_t)
            se_ev = math.sqrt(e_c.var(ddof=1) / nc + e_t.var(ddof=1) / nt)
            z_inv = (e_c.mean() - e_t.mean()) / se_ev
            # Гейт 3: guardrail авто-стоп — доля ошибок, порог +0.3 п.п.
            ec, et = np.concatenate(er_c), np.concatenate(er_t)
            pc, pt = ec.mean(), et.mean()
            pbar = (pc + pt) / 2
            se = math.sqrt(max(pbar * (1 - pbar), 1e-9) * (1 / nc + 1 / nt))
            z_gr = (pt - pc - 0.003) / se
            # OEC-путь «для иллюстрации» (платформа его прячет — peeking, 3.7)
            yc, yt = np.concatenate(y_c), np.concatenate(y_t)
            oec_cum = (yt.mean() - yc.mean()) / yc.mean()
            run.track(day, nc, nt, p_srm, z_inv, z_gr, pt - pc, oec_cum)

            if p_srm < srm_cutoff:
                run.stop(day, "SRM: расхождение выборок — отчёт заблокирован")
                break
            if abs(z_inv) > invariant_z:
                run.stop(day, "ИНВАРИАНТ: телеметрия разъехалась — стоп")
                break
            if z_gr > guardrail_z:
                run.stop(day, "GUARDRAIL AUTO-STOP: доля ошибок выросла значимо "
                              "выше порога +0.3 п.п. (fixed-K Bonferroni)")
                break
            if day == days:
                run.stop(day, "план выполнен")
        run.data = dict(y_c=np.concatenate(y_c), y_t=np.concatenate(y_t),
                        pre_c=np.concatenate(pre_c), pre_t=np.concatenate(pre_t),
                        err_c=np.concatenate(er_c), err_t=np.concatenate(er_t))
        return run

    # ---------------- analyze: OEC (Уэлч + CUPED) + вторичные (Холм) ----------------
    def analyze(self, ticket, run):
        d = run.data
        # CUPED: коэффициент по объединённым данным — large-n приближение.
        # Для точного ANCOVA adjustment используйте регрессию с T и robust SE (3.5–3.6).
        # CUPED по пре-периоду (урок 3.5): theta = cov(metric, pre)/var(pre)
        y_all = np.concatenate([d["y_c"], d["y_t"]])
        pre_all = np.concatenate([d["pre_c"], d["pre_t"]])
        theta = np.cov(y_all, pre_all)[0, 1] / np.var(pre_all, ddof=1)
        adj = lambda y, pre: y - theta * (pre - pre_all.mean())
        c, t = adj(d["y_c"], d["pre_c"]), adj(d["y_t"], d["pre_t"])
        stat = stats.ttest_ind(t, c, equal_var=False)          # Уэлч (урок 2.2)
        se = math.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
        diff = t.mean() - c.mean()
        oec = dict(est=diff, p=stat.pvalue,
                   rel=diff / c.mean(),
                   lo_rel=(diff - Z_A * se) / c.mean(),
                   hi_rel=(diff + Z_A * se) / c.mean(),
                   theta=theta, cuped_gain=1 - se/math.sqrt(d["y_t"].var(ddof=1)/len(t) + d["y_c"].var(ddof=1)/len(c)))

        # вторичные метрики (континуальные прокси, коррелированные с OEC)
        sec = {
            "выручка/юзер (винзор. P99)": self._sim_secondary(run, "rev"),
            "CTR карточки (нед.)": self._sim_secondary(run, "ctr"),
        }
        # Холм по возрастанию p (урок 2.4)
        names = list(sec)
        pvals = np.array([sec[k]["p"] for k in names])
        order = np.argsort(pvals)
        holm, running = {}, 0.0
        for rank, idx in enumerate(order):
            running = max(running, (len(pvals) - rank) * pvals[idx])
            holm[names[idx]] = min(running, 1.0)
        for k in names:
            sec[k]["holm_p"] = holm[k]

        # guardrail: доля юзеров с ошибкой (то, что могло остановить тест)
        nc, nt = len(d["err_c"]), len(d["err_t"])
        pc, pt = d["err_c"].mean(), d["err_t"].mean()
        se_g = math.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt)
        guardrail = dict(name="доля юзеров с ошибкой", diff=pt - pc,
                         breach=bool(pt - pc - stats.norm.isf(ALPHA)*se_g > 0.003),
                         safe=bool(pt - pc + stats.norm.isf(ALPHA)*se_g <= 0.003),
                         p=float(stats.norm.sf((pt - pc - 0.003) / se_g)))
        return dict(oec=oec, secondary=sec, guardrail=guardrail)

    @staticmethod
    def _sim_secondary(run, kind):
        """Вторичные метрики: генерируются поверх данных прогона."""
        d = run.data
        rng = np.random.default_rng(seed_of(run.exp_id + kind) % 2 ** 31)
        if kind == "rev":    # выручка = заказ × логнормальный чек
            chk = lambda y: y * rng.lognormal(6.65, 0.70, len(y))
            c, t = chk(d["y_c"]), chk(d["y_t"]) * 1.02
        else:                # CTR: +9% отн. за счёт механики
            c = rng.binomial(1, 0.576, len(d["y_c"]))
            t = rng.binomial(1, 0.576 * 1.09, len(d["y_t"]))
        w = lambda a: np.minimum(a, np.percentile(np.concatenate([c, t]), 99))
        cw, tw = w(c), w(t)                                     # винзоризация (3.4)
        st = stats.ttest_ind(tw, cw, equal_var=False)
        return dict(rel=(tw.mean() - cw.mean()) / cw.mean(), p=st.pvalue)

    # ---------------- verdict: decision rule 2x2 из урока 4.1 ----------------
    def verdict(self, ticket, run, an):
        if run.stop_reason.startswith(("SRM", "ИНВАРИАНТ")):
            return "БЛОК: проверка валидности не пройдена; итоговый эффект не интерпретируем"
        if run.stop_reason.startswith("GUARDRAIL"):
            return ("НЕ КАТИМ: авто-стоп по guardrail — разбор инцидента; "
                    "эффект не измерен до конца (обрыв — это не вердикт по OEC)")
        oec, gr = an["oec"], an["guardrail"]
        if not gr["safe"] and oec["est"] >= 0:
            return "HOLD: допустимый вред guardrail не исключён"
        if oec["p"] < ALPHA and oec["est"] > 0:
            return "SHIP: OEC значимо вырос, guardrails целы -> 50% -> 100%"
        if oec["p"] < ALPHA and oec["est"] < 0:
            return "ROLLBACK: OEC значимо упал"
        return "НЕТ РЕШЕНИЯ: оценка и CI не подтверждают пользу; следующий эксперимент планируется отдельно"

    def report(self, ticket, run, an, verdict_str):
        o, gr = an["oec"], an["guardrail"]
        sec_txt = "; ".join(f"{k.split(' (')[0]} {v['rel']:+.1%} "
                            f"(Holm p={v['holm_p']:.2f})"
                            for k, v in an["secondary"].items())
        return (f"{ticket['doc'].exp_id} «{ticket['doc'].name}» | стоп: день "
                f"{run.stop_day}/{run.planned_days} ({run.stop_reason.split(':')[0]})"
                f" | SRM p={run.p_srm_last:.2f} | OEC {o['rel']:+.1%} "
                f"(95% ДИ {o['lo_rel']:+.1%}..{o['hi_rel']:+.1%}, p={o['p']:.4f}; "
                f"CUPED сжал SE на {o['cuped_gain']:.0%}) | {sec_txt} | "
                f"guardrail {gr['name']} {gr['diff']:+.2%} (порог +0,3 п.п.) | "
                f"ВЕРДИКТ: {verdict_str}")


def monte_carlo_interval(successes, repetitions, confidence=0.95):
    """Clopper–Pearson interval for a frequency across independent simulations."""
    if repetitions <= 0 or not 0 <= successes <= repetitions:
        raise ValueError("Require 0 <= successes <= repetitions and repetitions > 0")
    tail = (1-confidence)/2
    lo = 0.0 if successes == 0 else stats.beta.ppf(tail, successes, repetitions-successes+1)
    hi = 1.0 if successes == repetitions else stats.beta.ppf(1-tail, successes+1, repetitions-successes)
    return float(lo), float(hi)


def loss_mechanism_demo(n=200_000, seed=73061):
    """True treatment effect is zero; two mechanisms have the same expected loss.

    Complete-case means illustrate bias from selecting on the outcome. This is
    separate from run's MCAR scenario and does not consume its random stream.
    """
    rng = np.random.default_rng(seed)
    p0, keep_zero = .23, .70
    assigned = rng.binomial(1, .5, n).astype(bool)
    outcome = rng.binomial(1, p0, n)
    control, treated = outcome[~assigned], outcome[assigned]
    keep_average = p0 + (1-p0)*keep_zero
    keep_mcar = rng.random(len(treated)) < keep_average
    keep_dependent = rng.random(len(treated)) < np.where(treated == 1, 1., keep_zero)
    rows = {}
    for name, keep in [('MCAR',keep_mcar),('outcome-dependent',keep_dependent)]:
        observed = treated[keep]
        diff = observed.mean()-control.mean()
        se = np.sqrt(observed.var(ddof=1)/len(observed)+control.var(ddof=1)/len(control))
        rows[name] = dict(estimate=float(diff), ci=(float(diff-1.96*se),float(diff+1.96*se)),
                          retained=int(keep.sum()), assigned=int(len(treated)),
                          srm_p=float(stats.chi2.sf((len(control)-len(observed))**2/(len(control)+len(observed)),1)))
    return dict(true_effect=0.0, expected_loss=1-keep_average,
                selected_limit=p0/keep_average-p0, rows=rows)


def calibrate_monitoring(repetitions=250, boundary=False, seed=73071):
    """Execute the actual whole stopping rule, then final OEC analysis if completed.

    SRM/invariant nulls hold, OEC effect=0. Guardrail is either interior (0
    difference) or on its null boundary (+.003). Secondary synthetic metrics are
    not null-calibrated here. No significance is interpreted after a gate stop.
    Intervals describe Monte Carlo error, not uncertainty about a real effect.
    """
    rng = np.random.default_rng(seed)
    platform = ExperimentPlatform()
    doc = DesignDoc('CALIBRATION','H0 calibration','интерфейс','конверсия в заказ (2 нед.)',.1)
    ticket = dict(doc=doc, weeks=2, p0=.23)
    counts = dict(SRM=0, invariant=0, guardrail=0, any_gate=0,
                  completed=0, final_oec_reject=0, false_ship=0)
    for _ in range(repetitions):
        run = platform.run(ticket, effect_rel=0., err_mult=1.15 if boundary else 1.,
                           rng_seed=int(rng.integers(0,2**31)))
        reason = run.stop_reason
        stopped = False
        for prefix,key in [('SRM','SRM'),('ИНВАРИАНТ','invariant'),('GUARDRAIL','guardrail')]:
            if reason.startswith(prefix):
                counts[key]+=1; counts['any_gate']+=1; stopped=True; break
        if not stopped:
            counts['completed']+=1
            analysis = platform.analyze(ticket,run)
            counts['final_oec_reject']+=int(analysis['oec']['p']<ALPHA)
            counts['false_ship']+=int(platform.verdict(ticket,run,analysis).startswith('SHIP'))
    return dict(repetitions=repetitions, boundary=boundary, planned_looks=14,
                any_gate_budget=2*SRM_ALPHA+ALPHA,
                counts=counts,
                rates={key:value/repetitions for key,value in counts.items()},
                intervals={key:monte_carlo_interval(value,repetitions) for key,value in counts.items()})


# %% [markdown]
# Шаг 2. Четыре заявки из бэклога «ЕдаДома». Платформа валидирует: считает n
# из MDE, подставляет guardrail-пакет, проверяет занятость слоёв. ED-119 —
# заведомо конфликтная (слой «ранжирование» занят на 50% живым тестом).

# %%
platform = ExperimentPlatform()
platform.LAYERS["ранжирование"].append(("ED-2026-101 идёт", 0.5))  # живой тест

doc1 = DesignDoc(
    exp_id="ED-2026-114", name="Умная сортировка по ETA",
    layer="ранжирование", oec="конверсия в заказ (2 нед.)", mde_rel=0.065,
    hyp_x="31% сессий не доходит до карточки", hyp_y="сортировка по ETA",
    hyp_z="заказы — драйвер выручки", share=0.5,
    secondary=("выручка/юзер (винзор. P99)", "CTR карточки (нед.)"))
doc2 = DesignDoc(
    exp_id="ED-2026-115", name="Пуш о брошенной корзине",
    layer="коммуникации", oec="открытие пуша (7 дн. окно)", mde_rel=0.16,
    hyp_x="72% корзин бросаются", hyp_y="пуш вернёт юзеров",
    hyp_z="доп. заказы без каннибализации")
doc3 = DesignDoc(
    exp_id="ED-2026-118", name="Новые иконки разделов",
    layer="интерфейс", oec="конверсия в заказ (2 нед.)", mde_rel=0.04,
    hyp_x="иконки не заметны", hyp_y="заметность -> CR", hyp_z="мягкий прирост CR")
doc4 = DesignDoc(  # конфликт: слой занят на 50%, просит 60%
    exp_id="ED-2026-119", name="Ранжирование с учётом рейтинга",
    layer="ранжирование", oec="конверсия в заказ (2 нед.)", mde_rel=0.05,
    share=0.6)

print("=" * 78)
print("1) SUBMIT: ЗАЯВКА = ДИЗАЙН-ДОК -> ВАЛИДАЦИЯ ПЛАТФОРМЫ")
print("-" * 78)
t1 = platform.submit(doc1)
t2 = platform.submit(doc2)
t3 = platform.submit(doc3)
t4 = platform.submit(doc4)  # отказ: пересечение в слое

# %% [markdown]
# Шаг 3. Прогон трёх сценариев: (1) чистый зелёный — истинный эффект +8% отн.
# при MDE 6,5%; (2) авто-стоп по guardrail — OEC растёт (+40%), но доля
# юзеров с ошибкой выросла ×1,6; (3) серый — истинный эффект +0,5% при MDE 4%.

# %%
print()
print("=" * 78)
print("2) RUN + GATES + ANALYZE + VERDICT: ТРИ СЦЕНАРИЯ")
print("-" * 78)

run1 = platform.run(t1, effect_rel=0.08)               # истинный эффект +8% отн.
an1 = platform.analyze(t1, run1)
print(platform.report(t1, run1, an1, platform.verdict(t1, run1, an1)))
print()

run2 = platform.run(t2, effect_rel=0.40, err_mult=1.6)  # OEC вверх, ошибки ×1,6
an2 = platform.analyze(t2, run2)
print(platform.report(t2, run2, an2, platform.verdict(t2, run2, an2)))
print()

run3 = platform.run(t3, effect_rel=0.005)              # истинный ~0
an3 = platform.analyze(t3, run3)
print(platform.report(t3, run3, an3, platform.verdict(t3, run3, an3)))

# %% [markdown]
# Шаг 4. Бонус: SRM-гейт как привратник. Тот же пуш, но телеметрия тестовой
# группы теряет 4,5% событий (кейс урока 4.2): платформа обязана остановить
# тест и заблокировать отчёт ДО того, как кто-то увидит «эффект».

# %%
print()
print("=" * 78)
print("3) SRM-ГЕЙТ: СЛОМАННЫЙ ТЕСТ ОСТАНАВЛИВАЕТСЯ ДО АНАЛИЗА")
print("   (Zalando: без автоматизации SRM молчал в >=20% тестов — ", sep="")
print("   автоматизация обязательна; Microsoft: тест без SRM не анализируется)")
print("-" * 78)
run4 = platform.run(t2, effect_rel=0.40, drop_test_share=0.045)
if run4.stop_reason.startswith("SRM"):
    nc, nt = run4.n_c[-1], run4.n_t[-1]
    chi2 = (nc - nt) ** 2 / (nc + nt)
    print(f"[{run4.exp_id}] день {run4.stop_day} из {run4.planned_days}: "
          f"N = {nc:,}/{nt:,} — разность размеров {nc - nt:,} "
          f"({100 * (nc - nt) / nc:.1f}% от контроля; это не точный счётчик потерь)")
    print(f"   хи-квадрат = {chi2:.1f}, p = {run4.p_srm_last:.1e} < {SRM_ALPHA/run4.planned_days:.2g} "
          f"-> СТОП, отчёт заблокирован до дебага (протокол урока 4.2)")
    an4 = platform.analyze(t2, run4)  # что показал бы дашборд БЕЗ гейта
    print(f"   а без гейта дашборд показал бы: OEC {an4['oec']['rel']:+.1%} "
          f"(p={an4['oec']['p']:.3f}) — реальный эффект уже задан; MCAR-потеря сама по себе не создаёт bias")
else:
    print(f"[{run4.exp_id}] неожиданно: стоп по другой причине — {run4.stop_reason}")

# %% [markdown]
# Шаг 5. Панель мониторинга платформы: SRM-гейт молчит «по построению»,
# guardrail-статистика пересекает порог — авто-стоп; OEC «спрятан» до конца
# (подглядывание — урок 3.7), для иллюстрации показан серым.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

ax = axes[0]
for run, c, lab in ((run1, "#55A868", "зелёный"), (run2, "#DD8452", "guardrail-стоп"),
                    (run3, "#4C72B0", "серый"), (run4, "#C44E52", "SRM-стоп")):
    ax.plot(run.days, run.p_srm, "-o", ms=3, lw=1.6, color=c, label=lab)
for idx, planned in enumerate(sorted({r.planned_days for r in (run1,run2,run3,run4)})):
    cutoff=SRM_ALPHA/planned
    ax.axhline(cutoff,color="#C44E52",lw=1,ls="--",alpha=.65)
    ax.text(1,cutoff*1.08,f"0,001/{planned} дней",color="#C44E52",fontsize=7)
ax.set_yscale("log")
ax.set_xlabel("день теста"); ax.set_ylabel("p SRM-чека (log)")
ax.set_title("Гейт 1: SRM молчит — не «всё ок»,\nа «проверка идёт каждый день»")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(run2.days, run2.z_gr, "-o", ms=3, lw=1.8, color="#DD8452", label="пуш (ошибки ×1,6)")
ax.plot(run1.days, run1.z_gr, "-", lw=1.4, color="#55A868", label="сортировка")
ax.plot(run3.days, run3.z_gr, "-", lw=1.4, color="#4C72B0", label="иконки")
gr_plot_z = stats.norm.isf(ALPHA / run2.planned_days)
ax.axhline(gr_plot_z, color="#C44E52", lw=1.8, ls="--")
ax.text(1, gr_plot_z + 0.15, f"граница пуша (K={run2.planned_days}): z = {gr_plot_z:.2f}", color="#C44E52", fontsize=9)
if run2.stopped_early and run2.stop_reason.startswith("GUARDRAIL"):
    ax.axvline(run2.stop_day, color="#C44E52", lw=1.4, ls=":")
    ax.annotate(f"авто-стоп, день {run2.stop_day}",
                (run2.stop_day, run2.z_gr[-1]), xytext=(-120, 10),
                textcoords="offset points", fontsize=9, color="#C44E52",
                arrowprops=dict(arrowstyle="->", color="#C44E52"))
ax.set_xlabel("день теста"); ax.set_ylabel("z: «доля ошибок выше порога +0,3 п.п.»")
ax.set_title("Гейт 2: авто-стоп по guardrail —\nпуш остановлен на полпути")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
for run, c, lab in ((run1, "#55A868", "зелёный"), (run3, "#4C72B0", "серый")):
    ax.plot(run.days, 100 * np.array(run.oec_path), "-", lw=1.8, color=c, label=lab)
ax.axhline(0, color="gray", lw=1)
ax.set_xlabel("день теста"); ax.set_ylabel("накопленный эффект OEC, % отн.")
ax.set_title("OEC: платформа прячет до конца теста\n(peeking — урок 3.7); в конце — вердикт 4.1")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("Мини-платформа «ЕдаДома»: заявка -> гейты -> анализ -> вердикт", y=1.03)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_7_3_monitoring.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nСохранено: practice_7_3_monitoring.png")

# %%
print()
print("=" * 78)
print("ГЛАВНОЕ (что должны увидеть):")
print("1) Заявка = дизайн-док: платформа сама считает n из MDE, подставляет")
print("   guardrail-пакет и отклоняет ED-119 — слой переполнен (очередь — 7.4).")
print("2) Гейты автоматические: SRM каждый день (молчание = проверка идёт; без")
print("   автоматизации SRM молчал в >=20% тестов у Zalando); авто-стоп по")
print("   guardrail останавливает пуш на полпути — «зелёный OEC» не спасает.")
print("3) Анализ: OEC = Уэлч + CUPED (сжатие SE), вторичные — с Холмом;")
print("   вердикт — по decision rule 2x2 из 4.1, а не «по ощущениям».")
print("4) Серый тест — валидный исход: «нет решения», а не «эффекта нет».")

# Separate seeds leave the four original scenarios unchanged.
print("\n4) ДВА МЕХАНИЗМА ПОТЕРЬ ПРИ ИСТИННОМ ЭФФЕКТЕ 0")
loss_demo = loss_mechanism_demo()
print(f"Одинаковая ожидаемая потеря {loss_demo['expected_loss']:.1%}; complete-case оценки:")
for name,row in loss_demo['rows'].items():
    print(f"  {name}: Δ={100*row['estimate']:+.3f} п.п., 95% CI [{100*row['ci'][0]:+.3f}; {100*row['ci'][1]:+.3f}] п.п., "
          f"осталось {row['retained']}/{row['assigned']}")
print(f"  Outcome-dependent limit = {100*loss_demo['selected_limit']:+.3f} п.п.; SRM обнаруживает дисбаланс, но не объясняет механизм.")
print("\n5) H0-КАЛИБРОВКА ВСЕЙ ПРОЦЕДУРЫ: 14 ПЛАНОВЫХ ПРОСМОТРОВ")
for boundary in [False,True]:
    calibration = calibrate_monitoring(boundary=boundary,seed=73071+int(boundary))
    print(f"  Guardrail true Δ={'0.003 (граница H0)' if boundary else '0 (внутри H0)'}; "
          f"B={calibration['repetitions']}; union budget любого гейта={calibration['any_gate_budget']:.3f}")
    for key in ['SRM','invariant','guardrail','any_gate','final_oec_reject','false_ship']:
        lo,hi=calibration['intervals'][key]
        print(f"    {key}: {calibration['counts'][key]}/{calibration['repetitions']}="
              f"{calibration['rates'][key]:.3%}; Monte Carlo 95% CI [{lo:.3%}; {hi:.3%}]")
print("  Знаменатель частот — все симуляции; финальный OEC проверяется только без gate-stop.")
print("  Нулевой счётчик не означает нулевой риск. Bonferroni требует корректных fixed-look p;")
print("  normal/chi-square приближения проверены только для этого DGP, не для любой телеметрии.")
print("  B=250 даёт грубую оценку редких SRM/invariant ошибок уровня 0,001; для точности нужно больше миров.")
