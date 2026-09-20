# -*- coding: utf-8 -*-
"""Практика 7.1 — Сплит-инфраструктура платформы «ЕдаДома»
(урок 7.1 модуля «Проектирование АБ-платформы»).

Принцип курса: не верь формуле — проверь симуляцией.

Что делаем:
1) Лаборатория хэшей: MD5 против FNV-1a на ПОСЛЕДОВАТЕЛЬНЫХ id (самый частый
   случай в проде). Равномерность, лавинный эффект, независимость солей —
   эмпирическая проверка вывода Microsoft (arXiv 2212.08771): крипто- и
   качественные некрипто-хэши дают независимые сплиты, FNV — нет;
2) Сплит-сервис как класс: домены -> слои -> 1000 бакетов (Google KDD'10),
   версии конфига, first-fit аллокация слотов, конфликт-детектор,
   отчёт о ёмкости слоёв + цена слота в MDE (N_total >= 4*(z_alpha/2+z_power)^2*(s/theta)^2);
3) Sticky assignment: миграция ключа cookie -> account посреди теста
   (что видит юзер при наивном переключении) и cookie churn ~4%/мес
   (Deng, гл.7) — с персистом и без.

Запуск из корня репозитория: python3 course/modules/M7/practice/practice_7_1.py
Графики сохраняются в папку images/ этого модуля.
"""

# %% [markdown]
# Шаг 0. Импорты и параметры. Сплит «ЕдаДома» строим в стиле Google KDD'10:
# 1000 бакетов на слой, слои сидят в доменах (доля трафика), у каждого слоя
# своя соль. Хэш-функция по умолчанию — MD5 (независимость доказана MS).

# %%
from __future__ import annotations

import copy
import hashlib
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260913)
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

N_BUCKETS = 1_000          # бакетов в слое (канон Google KDD'10)
FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3


def md5_int(data: bytes) -> int:
    return int(hashlib.md5(data).hexdigest(), 16)


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64 — быстрый, но слабый на диффузию некрипто-хэш."""
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def get_bucket(unit_key: str, salt: str, algo: str = "md5", m: int = N_BUCKETS) -> int:
    payload = f"{salt}|{unit_key}".encode()
    h = md5_int(payload) if algo == "md5" else fnv1a64(payload)
    return h % m


# %% [markdown]
# Шаг 1. Лаборатория хэшей. Проверяем три требования (4.2 — определение,
# здесь — выбор функции): детерминизм очевиден, смотрим равномерность на
# ПОСЛЕДОВАТЕЛЬНЫХ id, лавинность (соседние id) и независимость двух солей.

# %%
users = [f"user_{i:07d}" for i in range(20_000)]

# (а) лавинный эффект: id, отличающийся на одну цифру, должен переворачивать
# ~половину выходных бит — иначе похожие id попадают в связанные бакеты
ham_md5, ham_fnv = [], []
for i in range(2_000):
    a, b = f"user_{i:07d}".encode(), f"user_{i + 1:07d}".encode()
    ham_md5.append(bin(md5_int(b"layer|" + a) ^ md5_int(b"layer|" + b)).count("1"))
    ham_fnv.append(bin(fnv1a64(b"layer|" + a) ^ fnv1a64(b"layer|" + b)).count("1"))
ham_md5, ham_fnv = np.array(ham_md5), np.array(ham_fnv)

# (б) вероятность переворота КАЖДОГО бита: у хорошего хэша все ~0.5
FLIP_N = 1_000
flip_md5 = np.zeros(128)
flip_fnv = np.zeros(64)
for i in range(FLIP_N):
    a, b = f"user_{i:07d}".encode(), f"user_{i + 1:07d}".encode()
    x = md5_int(b"layer|" + a) ^ md5_int(b"layer|" + b)
    for k in range(128):
        flip_md5[k] += (x >> k) & 1
    y = fnv1a64(b"layer|" + a) ^ fnv1a64(b"layer|" + b)
    for k in range(64):
        flip_fnv[k] += (y >> k) & 1
flip_md5 /= FLIP_N
flip_fnv /= FLIP_N

# (в) равномерность 1000 бакетов + (г) независимость двух солей (10x10)
SALT_U = "salt_ranking"   # при этой соли провал FNV виден в лоб
b_md5 = np.array([get_bucket(u, SALT_U) for u in users])
b_fnv = np.array([get_bucket(u, SALT_U, algo="fnv") for u in users])
b_md5_other = np.array([get_bucket(u, "layer_ranking") for u in users])
b_fnv_other = np.array([get_bucket(u, "layer_ranking", algo="fnv") for u in users])


def chi2_uniform(buckets, m):
    c = np.bincount(buckets, minlength=m).astype(float)
    e = len(buckets) / m
    return float(((c - e) ** 2 / e).sum()), stats.chi2.sf(((c - e) ** 2 / e).sum(), m - 1)


chi2_m, p_m = chi2_uniform(b_md5, N_BUCKETS)
chi2_f, p_f = chi2_uniform(b_fnv, N_BUCKETS)
chi2_mo, p_mo = chi2_uniform(b_md5_other, N_BUCKETS)
chi2_fo, p_fo = chi2_uniform(b_fnv_other, N_BUCKETS)

tab_m = np.zeros((10, 10))
tab_f = np.zeros((10, 10))
for u in users:
    tab_m[get_bucket(u, "s1") // 100, get_bucket(u, "s2") // 100] += 1
    tab_f[get_bucket(u, "s1", algo="fnv") // 100, get_bucket(u, "s2", algo="fnv") // 100] += 1
chi2_im, p_im, _, _ = stats.chi2_contingency(tab_m)
chi2_if, p_if, _, _ = stats.chi2_contingency(tab_f)

print("=" * 78)
print(f"1) ЛАБОРАТОРИЯ ХЭШЕЙ: MD5 vs FNV-1a на {len(users):,} последовательных id")
print("-" * 78)
print(f"   лавинный эффект (соседние id):  MD5 {ham_md5.mean():.1f}/128 бит (идеал 64.0)"
      f"   |   FNV {ham_fnv.mean():.1f}/64 бит (идеал 32.0)")
print(f"   переворот битов: MD5 min..max = {flip_md5.min():.2f}..{flip_md5.max():.2f}"
      f"   |   FNV min..max = {flip_fnv.min():.2f}..{flip_fnv.max():.2f}")
worst = int(np.argmax(np.abs(flip_fnv - 0.5)))
print(f"   худший бит FNV: #{worst} переворачивается с вероятностью {flip_fnv[worst]:.2f}"
      f" (нужно 0.50) — а именно младшие биты идут в bucket % 1000")
print(f"   равномерность 1000 бакетов:  MD5 chi2={chi2_m:.0f}, p={p_m:.2f}"
      f"   |   FNV chi2={chi2_f:.0f}, p={p_f:.0e} -> НЕравномерно")
print(f"   та же проверка с другой солью ('layer_ranking'): MD5 p={p_mo:.2f}; FNV p={p_fo:.2f} — ПРОШЛА")
print(f"   независимость двух солей:    MD5 chi2={chi2_im:.0f}, p={p_im:.2f}"
      f"   |   FNV chi2={chi2_if:.0f}, p={p_if:.4f} -> корреляция сплитов")
print("""   Вывод: FNV-1a ломается НЕПРЕДСКАЗУЕМО: с одной солью равномерность валится
   (p~1e-34), с другой проходит (p=1.0) — а независимость слоёв падает всё равно.
   То же, что у Microsoft (arXiv 2212.08771): SpookyHash/MD5 ок, FNV нет. Соль не
   лечит: она меняет отображение, но не слабую диффузию младших бит.""")

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
ax = axes[0]
ax.hist(ham_md5, bins=24, alpha=0.7, color="#4C72B0", label=f"MD5: {ham_md5.mean():.1f}/128")
ax.hist(ham_fnv, bins=24, alpha=0.7, color="#C44E52", label=f"FNV-1a: {ham_fnv.mean():.1f}/64")
ax.axvline(128 * 0.5, color="#4C72B0", ls=":", lw=1.5)
ax.axvline(64 * 0.5, color="#C44E52", ls=":", lw=1.5)
ax.set_xlabel("сколько бит изменилось у соседних id")
ax.set_ylabel("пар id")
ax.set_title("Лавинный эффект: FNV меняет ~10 из 64 бит —\nпохожие id остаются связанными")
ax.legend(fontsize=9)
ax = axes[1]
ax.bar(np.arange(64), np.abs(flip_fnv - 0.5), color="#C44E52", alpha=0.85, label="FNV-1a")
ax.bar(np.arange(64, 128), np.abs(flip_md5[:64] - 0.5), color="#4C72B0", alpha=0.85, label="MD5 (первые 64 бита)")
ax.axhline(0.5, color="gray", ls="--", lw=1, label="порог катастрофы")
ax.set_xlabel("номер выходного бита (FNV: 0–63, MD5: 64–127)")
ax.set_ylabel("|P(переворота) − 0.5|")
ax.set_title("Диффузия по битам: у FNV младшие биты прикованы\n(а % 1000 берёт именно их)")
ax.legend(fontsize=8)
ax = axes[2]
c_m = np.bincount(b_md5, minlength=N_BUCKETS)
c_f = np.bincount(b_fnv, minlength=N_BUCKETS)
ax.plot(c_m, lw=0.4, color="#4C72B0", label=f"MD5: p(хи-квадрат)={p_m:.2f}")
ax.plot(c_f, lw=0.4, color="#C44E52", alpha=0.8, label=f"FNV: p={p_f:.0e}")
ax.axhline(20, color="gray", ls="--", lw=1.2, label="ожидание 20")
ax.set_xlabel("бакет (0..999)")
ax.set_ylabel("юзеров")
ax.set_title("Равномерность 1000 бакетов на последовательных id:\nMD5 — ровно, FNV — структурные горбы")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_7_1_hash.png", dpi=150)
plt.close(fig)
print("\nСохранено: practice_7_1_hash.png")

# %% [markdown]
# Шаг 2. Сплит-сервис «ЕдаДома»: домены -> слои -> 1000 бакетов (KDD'10).
# Конфиг версионируется (каждая мутация = новый номер): экспозиция несёт
# версию, анализ воспроизводится по снапшоту. Аллокация слотов — first-fit
# по свободным диапазонам бакетов; конфликт ловится ДО запуска теста.

# %%
class AllocationConflict(Exception):
    """Нет свободного непрерывного диапазона бакетов нужного размера."""


class SplitService:
    """Мини-сплит-сервис в стиле Google KDD'10 / Microsoft ExP EES."""

    def __init__(self, n_buckets: int = N_BUCKETS):
        self.n_buckets = n_buckets
        self.domains: dict[str, dict] = {}   # домен: доля трафика + список слоёв
        self.layers: dict[str, dict] = {}    # слой: соль + [(имя, доля, варианты)]
        self.version = 0
        self.change_log: list[str] = []

    # -- устройство -----------------------------------------------------
    def add_domain(self, name: str, traffic: float = 1.0) -> None:
        if not 0 < traffic <= 1:
            raise ValueError("traffic должен быть в (0, 1]")
        self.domains[name] = {"traffic": traffic, "layers": [], "salt": f"domain_{name}"}
        self._bump(f"домен {name} ({traffic:.0%} трафика)")

    def add_layer(self, name: str, domain: str, salt: str | None = None) -> None:
        self.layers[name] = {"domain": domain, "salt": salt or f"salt_{name}", "alloc": []}
        self.domains[domain]["layers"].append(name)
        self._bump(f"слой {name} в домене {domain}, соль {self.layers[name]['salt']}")

    def _bump(self, msg: str) -> None:
        self.version += 1
        self.change_log.append(f"v{self.version}: {msg}")

    def snapshot(self) -> dict:
        """Неизменяемая копия конфига — её несёт каждая экспозиция."""
        return copy.deepcopy({"n_buckets": self.n_buckets, "layers": self.layers,
                              "domains": self.domains, "version": self.version})

    # -- аллокация и конфликты -------------------------------------------
    def free_ranges(self, layer: str) -> list[tuple[int, int]]:
        busy = sorted((a, a + size) for _, a, size, _ in self.layers[layer]["alloc"])
        free, pos = [], 0
        for a, b in busy:
            if a > pos:
                free.append((pos, a))
            pos = max(pos, b)
        if pos < self.n_buckets:
            free.append((pos, self.n_buckets))
        return free

    def add_experiment(self, layer: str, name: str, share: float, n_variants: int = 2) -> tuple[int, int]:
        if any(name == existing for config in self.layers.values()
               for existing, _, _, _ in config["alloc"]):
            raise ValueError(f"Эксперимент {name} уже существует; ID должен быть уникальным")
        if not 0 < share <= 1 or n_variants < 2:
            raise ValueError("Некорректная доля или число вариантов")
        need = round(share * self.n_buckets)
        if need < n_variants:
            raise ValueError("Недостаточно бакетов для вариантов")
        for a, b in self.free_ranges(layer):
            if b - a >= need:  # first-fit
                self.layers[layer]["alloc"].append((name, a, need, n_variants))
                self._bump(f"{layer}: эксперимент {name} на {share:.0%} (бакеты {a}..{a + need - 1})")
                return a, a + need
        max_free = max((b - a for a, b in self.free_ranges(layer)), default=0)
        taken = ", ".join(f"{n}={s / self.n_buckets:.0%}" for n, _, s, _ in self.layers[layer]["alloc"])
        raise AllocationConflict(
            f"{layer}: нужно {need} бакетов ({share:.0%}), макс. свободный непрерывный"
            f" диапазон {max_free} бакетов ({max_free / self.n_buckets:.0%})."
            f" Занято: {taken or '—'}")

    # -- назначение -------------------------------------------------------
    def assign(self, layer: str, user_id: str | None = None, cookie: str | None = None,
               date: str | None = None, algo: str = "md5") -> str | None:
        """Независимые хэши допуска в домен и назначения внутри слоя.

        Без устойчивого ключа не включаем пользователя. cookie+date означает
        дневной юнит и допустим только для явно спроектированного такого дизайна.
        """
        key = user_id if user_id is not None else cookie
        if key is None:
            return None
        if user_id is None and date is not None:
            key = f"{cookie}|{date}"
        config = self.layers[layer]
        domain = self.domains[config["domain"]]
        if get_bucket(key, domain["salt"], algo, 1_000_000) >= domain["traffic"]*1_000_000:
            return None
        bucket = get_bucket(key, config["salt"], algo, self.n_buckets)
        for name, a, size, n_var in config["alloc"]:
            if a <= bucket < a + size:
                v = (bucket-a)*n_var//size
                return f"{name}:v{v}" if n_var > 2 else f"{name}:{'A' if v == 0 else 'B'}"
        return None

    @classmethod
    def from_snapshot(cls, snapshot):
        """Восстанавливает полный конфиг для воспроизводимого replay."""
        service = cls(snapshot["n_buckets"])
        service.layers = copy.deepcopy(snapshot["layers"])
        service.domains = copy.deepcopy(snapshot["domains"])
        service.version = snapshot["version"]
        return service

    def capacity_report(self) -> pd.DataFrame:
        rows = []
        for lname, l in self.layers.items():
            used = sum(size for _, _, size, _ in l["alloc"])
            rows.append({"слой": lname, "домен": l["domain"], "соль": l["salt"],
                         "занято": f"{used / self.n_buckets:.0%}",
                         "экспериментов": len(l["alloc"]),
                         "макс. непрерывный слот": f"{max((b - a for a, b in self.free_ranges(lname)), default=0) / self.n_buckets:.0%}"})
        return pd.DataFrame(rows)


# %%
svc = SplitService()
svc.add_domain("search", traffic=1.0)
svc.add_domain("logistics", traffic=0.5)          # домен = доля трафика
svc.add_layer("ranking", "search", salt="layer_ranking_v1")
svc.add_layer("ui", "search", salt="layer_ui_v1")
svc.add_layer("courier_algo", "logistics", salt="layer_courier_v1")

v_before = svc.version
snap = svc.snapshot()                              # анализ будет реплеить по нему

for index, share in enumerate((0.50, 0.25, 0.10, 0.10, 0.05), start=1):
    svc.add_experiment("ranking", f"rank_test_{index}_{int(share * 100)}", share)
svc.add_experiment("ui", "big_banner", 0.60)
svc.add_experiment("ui", "icon_set", 0.30)

print("=" * 78)
print("2) СПЛИТ-СЕРВИС: домены -> слои -> 1000 бакетов, first-fit аллокация")
print("-" * 78)
print(svc.capacity_report().to_string(index=False))
print(f"\n   версии конфига: было v{v_before}, стало v{svc.version}"
      f" — экспозиции несут версию, снапшот v{v_before} неизменен:"
      f" {snap['layers']['ranking']['alloc'] == []}")

# конфликт-детектор: просим 10% в слой ranking, где свободно 0%
try:
    svc.add_experiment("ranking", "rank_test_conflict", 0.10)
except AllocationConflict as e:
    print(f"\n   КОНФЛИКТ ПОЙМАН ДО ЗАПУСКА: {e}")
# и фрагментация: в ui свободно 10%, но просим непрерывные 10% после 60+30 —
# свободный диапазон ровно 100 бакетов: 10% влезает впритык, 15% — нет
svc.add_experiment("ui", "checkout_hint", 0.10)
try:
    svc.add_experiment("ui", "one_more", 0.15)
except AllocationConflict as e:
    print(f"   ФРАГМЕНТАЦИЯ: {e}")

# назначение: приоритет user_id -> cookie (KDD'10); слои независимы по солям
demo = [("u_000111", None), (None, "ck_777")]
for uid, ck in demo:
    print(f"   assign(ranking, user_id={uid}, cookie={ck}) -> "
          f"{svc.assign('ranking', user_id=uid, cookie=ck)}; "
          f"assign(ui, ...) -> {svc.assign('ui', user_id=uid, cookie=ck)}")

# %% [markdown]
# Ёмкость и цена слота. Формула KDD'10 для 95%/80%: N_total >= 4*(z_alpha/2+z_power)^2*(s/theta)^2.
# Слот доли f даёт MDE в 1/sqrt(f) раз больше (урок 3.1) и длится в 1/f раз
# дольше. Метрика «ЕдаДома»: GMV/юзера за 2 недели, CV ~ 2.5 (логнормальная).

# %%
def mde_for_total(n_total, cv, alpha=0.05, power=0.8):
    """Относительный MDE: общий N, равные руки, нормальная аппроксимация."""
    if n_total <= 0 or cv < 0 or not 0 < alpha < 1 or not 0.5 < power < 1:
        raise ValueError("Проверьте N, CV, alpha и power")
    return 2 * (stats.norm.isf(alpha / 2) + stats.norm.ppf(power)) * cv / math.sqrt(n_total)


CV = 2.5                       # коэффициент вариации GMV на юзера
USERS_2WK = 2_000_000          # активных юзеров за 2 недели на «ЕдаДома»
rows = []
for share in (1.00, 0.50, 0.25, 0.10, 0.05):
    n_total = USERS_2WK * share
    mde = mde_for_total(n_total, CV)
    rows.append({"доля слоя": f"{share:.0%}",
                 "параллельных тестов такого размера (на 1 слой)": int(1 / share),
                 "MDE (отн.)": f"{mde:.2%}",
                 "MDE × к 100%": f"{math.sqrt(1 / share):.2f}",
                 "недель на точность 100%-теста": 1 / share})
df_cap = pd.DataFrame(rows)
print("\n   Ёмкость 1 слоя = 1000 бакетов; цена слота — в MDE и длительности:")
print(df_cap.to_string(index=False))
print(f"""   Проверка на 100%: MDE = sqrt(31.4*CV^2/N_total) = {mde_for_total(USERS_2WK, CV):.2%}
   Сценарий «ЕдаДома»: 6 слоёв (ranking, ui, 2x checkout, courier, CRM) дают
   {6 * 4} параллельных теста по 25% или {6 * 20} по 5% — календарь гипотез (7.4).""")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
# карта занятости слоёв: кто где сидит
for row_i, lname in enumerate(["ranking", "ui", "courier_algo"]):
    for name, a, size, _ in svc.layers[lname]["alloc"]:
        ax1.barh(row_i, size, left=a, height=0.55,
                 color=plt.cm.tab20(hash(name) % 20), alpha=0.9, edgecolor="white")
        if size >= 100:
            ax1.text(a + size / 2, row_i, f"{name}\n{size / 10:.0f}%", ha="center",
                     va="center", fontsize=7)
ax1.set_yticks([0, 1, 2], ["ranking\n(search)", "ui\n(search)", "courier_algo\n(logistics 50%)"])
ax1.set_xlabel("бакеты 0..999")
ax1.set_xlim(0, N_BUCKETS)
ax1.set_title("Карта занятости слоёв: каждый слой — свои 1000 бакетов;\nсвободное = слоты для следующих тестов")
ax2.plot([5, 10, 25, 50, 100], [math.sqrt(100 / s) for s in (5, 10, 25, 50, 100)], "o-", color="#4C72B0")
for s in (5, 10, 25, 50, 100):
    ax2.annotate(f"×{math.sqrt(100 / s):.2f}", (s, math.sqrt(100 / s)), xytext=(4, 6),
                 textcoords="offset points", fontsize=9)
ax2.set_xscale("log")
ax2.set_xticks([5, 10, 25, 50, 100], ["5%", "10%", "25%", "50%", "100%"])
ax2.set_xlabel("доля слоя, выделенная тесту")
ax2.set_ylabel("MDE относительно теста на 100%")
ax2.set_title("Цена слота: тест на 5% трафика видит эффект\nв 4.47 раза крупнее (или длится ×20)")
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_7_1_capacity.png", dpi=150)
plt.close(fig)
print("Сохранено: practice_7_1_capacity.png")

# %% [markdown]
# Шаг 3. Sticky assignment: где жить назначению. А) Миграция ключа
# cookie -> account посреди теста: наивное переключение = полная
# ре-рандомизация (лавинный эффект); лечится переходным персистом.
# Б) Cookie churn ~4%/мес (Deng, гл.7): без привязки к аккаунту юзеры
# «мигрируют» между группами и размывают эффект.

# %%
N_USERS = 120_000
has_acc = RNG.random(N_USERS) < 0.35              # 35% залогинены
cookies = [f"ck_{i}" for i in range(N_USERS)]
accounts = [f"acc_{RNG.integers(0, 10 * N_USERS)}" if h else None for h in has_acc]
salt = "layer_ranking_v1"

# тест идёт по cookie (анонимы+логины) — вариант на старте
var_start = np.array([get_bucket(c, salt) // 500 % 2 for c in cookies])  # 50/50 A/B

# (А) платформа переводит слой на account_id для залогиненных СЕЙЧАС
var_after_naive = np.where(
    has_acc,
    [get_bucket(a, salt) // 500 % 2 if a else v for a, v in zip(accounts, var_start)],
    var_start)
switched = has_acc & (var_after_naive != var_start)
match_naive = float(np.mean(var_after_naive[has_acc] == var_start[has_acc]))

# (Б) правильная миграция: для переходной когорты персистим старое назначение
# (карта cookie -> вариант до конца теста), новые юзеры — уже по account
sticky_map = {c: v for c, v in zip(cookies, var_start)}
var_after_sticky = np.array([sticky_map[c] for c in cookies])

print("=" * 78)
print(f"3) STICKY ASSIGNMENT: миграция ключа и cookie churn, {N_USERS:,} юзеров")
print("-" * 78)
print(f"   залогинены (есть account_id): {has_acc.mean():.0%}")
print(f"   наивная миграция на account: вариант сохранили {match_naive:.1%} юзеров —"
      f" {switched.mean():.1%} всей базы сменили группу")
print(f"   это НЕ баг хэша, а его свойство (лавинность): смена ключа = ре-рандомизация")
print(f"   миграция со sticky-персистом переходной когорты: сохранили "
      f"{np.mean(var_after_sticky == var_start):.0%} — цена = хранение карты до конца теста")

# (В) cookie churn: ~4%/мес чистка кук / смена устройства; тест 4 недели
TRUE_LIFT = 0.05                       # истинный эффект B: +5% GMV
base_gmv = RNG.lognormal(mean=math.log(60), sigma=0.9, size=N_USERS)
gmv0 = base_gmv * (1 + TRUE_LIFT * var_start)          # мир без churn

p_reset_week = 0.04 / 4                                # ~4%/мес -> 1%/нед
wk_reset = RNG.random((4, N_USERS)) < p_reset_week
affected = wk_reset.any(axis=0)
# у пострадавших новый cookie -> перевыбор варианта с шансом 50%
new_var = np.where(RNG.random(N_USERS) < 0.5, 1 - var_start, var_start)
# на аккаунте churn ниже: смена устройства не меняет ключ (0.2%/мес)
acc_reset_week = 0.002 / 4
wk_acc = RNG.random((4, N_USERS)) < acc_reset_week
acc_affected = wk_acc.any(axis=0)
new_var_acc = np.where(RNG.random(N_USERS) < 0.5, 1 - var_start, var_start)

obs_cookie = np.where(affected, base_gmv * (1 + TRUE_LIFT * new_var), gmv0)
obs_account = np.where(acc_affected, base_gmv * (1 + TRUE_LIFT * new_var_acc), gmv0)
# наблюдаемые эффекты считаем по СТАРТОВОЙ группе (анализ не знает о смене);
# референс — lift того же сэмпла БЕЗ churn (чтобы убрать шум одной реализации)
lift0 = gmv0[var_start == 1].mean() / gmv0[var_start == 0].mean() - 1
lift_cookie = obs_cookie[var_start == 1].mean() / obs_cookie[var_start == 0].mean() - 1
lift_account = obs_account[var_start == 1].mean() / obs_account[var_start == 0].mean() - 1
print(f"\n   cookie churn 4%/мес за 4 недели: затронуто {affected.mean():.1%} юзеров,")
print(f"   сменили группу ~{(affected & (new_var != var_start)).mean():.1%}")
print(f"   наблюдаемый эффект: без churn {lift0:.2%}; по cookie-ключу {lift_cookie:.2%}"
      f" (размытие {(lift0 - lift_cookie) * 100:.2f} п.п.);"
      f" по account-ключу {lift_account:.2%}")
print("""   Мораль: персист «в cookie» бесплатен, пока ключ не меняется; смена ключа
   (cookie->account, миграция ID-пространства) — это миграция теста: переходное
   окно со sticky-картой, иначе половина базы меняет группу посреди теста.""")

# %%
weeks = np.arange(1, 5)
dilution = []
cum = np.zeros(N_USERS, dtype=bool)
for w in range(4):
    cum |= wk_reset[w]
    switched_w = cum & (new_var != var_start)
    gmv_w = np.where(switched_w, base_gmv * (1 + TRUE_LIFT * new_var), gmv0)
    dilution.append(gmv_w[var_start == 1].mean() / gmv_w[var_start == 0].mean() - 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
ax1.bar([0, 1], [match_naive, 1.0], color=["#C44E52", "#55A868"], width=0.5)
ax1.set_xticks([0, 1], ["наивная миграция\nна account", "миграция со sticky-\nперсистом когорты"])
for x, v in ((0, match_naive), (1, 1.0)):
    ax1.text(x, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
ax1.set_ylim(0, 1.15)
ax1.set_ylabel("доля юзеров, сохранивших вариант")
ax1.set_title("Смена ключа хэша = ре-рандомизация:\nперсист лечит, хэш — нет")
ax2.plot(weeks, np.array(dilution) * 100, "o-", color="#4C72B0", label="ключ = cookie")
ax2.axhline(lift0 * 100, color="#55A868", ls="--", lw=2,
            label=f"без churn: lift сэмпла {lift0:.2%}")
ax2.plot(weeks, np.full(4, lift_account * 100),
         "s--", color="#8172B3", label="ключ = account (churn 0.2%/мес)")
ax2.set_xlabel("неделя теста")
ax2.set_ylabel("наблюдаемый эффект, %")
ax2.set_title("Cookie churn ~4%/мес размывает эффект:\n~2% базы тихо меняют группу за месяц")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMAGE_DIR / "practice_7_1_sticky.png", dpi=150)
plt.close(fig)
print("Сохранено: practice_7_1_sticky.png")
print("\nГотово: хэш-лаборатория -> сплит-сервис -> sticky. Дальше (7.2) — куда")
print("этот сплит пишет данные и сколько они стоят.")
