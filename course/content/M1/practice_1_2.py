# %% [markdown]
# # Практика 1.2 — Зоопарк распределений метрик «ЕдаДома»
#
# Метрики кейса: конверсия в первый заказ (бинарная), заказы в час (счётчик),
# время доставки (непрерывная, аддитивная), чек (деньги, мультипликативная).
#
# План:
# 1. Зоопарк: 4 распределения продуктовых метрик + теоретические кривые.
# 2. Сколько бинов в гистограмме: Стерджесс / Скотт / Фридман–Диаконис.
# 3. MoM vs MLE для Пуассона (число заказов) + ловушка overdispersion.
# 4. MoM vs MLE для логнормалы (чек) — здесь они расходятся.
# 5. ECDF vs гистограмма на времени доставки.
# 6. QQ-plot логнормального чека: сырой vs логарифмированный.
#
# Культура курса: не верь формуле — проверь симуляцией и графиком.

# %%
import matplotlib

matplotlib.use("Agg")  # сохраняем png без экрана
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

plt.rcParams["figure.dpi"] = 110
HERE = Path(__file__).resolve().parent  # png лягут рядом со скриптом

rng = np.random.default_rng(2027)  # один генератор на весь ноутбук

# %% [markdown]
# ## 1. Зоопарк распределений продуктовых метрик
#
# - конверсия в первый заказ: число конверсий в когорте ~ Binomial(n=1000, p=0.12);
# - заказы в час на район ~ Poisson(λ=40);
# - время доставки ~ Normal(μ=34 мин, σ=6 мин);
# - чек ~ Lognormal(median=900 ₽, σ=0.55).
#
# На каждый — симуляция + теоретическая PMF/PDF. Смотрим, как теория ложится.

# %%
N_COHORTS, P_CONV = 500, 0.12      # 500 дневных когорт по 1000 новых юзеров
LAMBDA_HOUR = 40                   # заказов в час на район
MU_T, SIGMA_T = 34.0, 6.0          # время доставки, мин
MEDIAN_CHECK, SIGMA_CHECK = 900.0, 0.55  # чек: медиана и σ логнормалы

conversions = rng.binomial(1000, P_CONV, size=N_COHORTS)
orders_hour = rng.poisson(LAMBDA_HOUR, size=2000)
delivery_min = rng.normal(MU_T, SIGMA_T, size=3000)
check = rng.lognormal(mean=np.log(MEDIAN_CHECK), sigma=SIGMA_CHECK, size=5000)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# --- биномиальное: конверсия ---
ax = axes[0, 0]
ks = np.arange(conversions.min(), conversions.max() + 1)
ax.hist(conversions, bins=np.arange(ks[0] - 0.5, ks[-1] + 1.5, 1),
        density=True, color="#4C72B0", edgecolor="white", linewidth=0.4,
        label="выборка (500 когорт)")
ax.plot(ks, stats.binom.pmf(ks, 1000, P_CONV), "o-", color="#C44E52",
        ms=3, lw=1, label="теория Binom(1000; 0.12)")
ax.set_title(f"Конверсия в 1-й заказ: Binom(n=1000, p={P_CONV})\n"
             f"sd = √(np(1-p)) = {np.sqrt(1000 * P_CONV * (1 - P_CONV)):.1f} конверсий")
ax.set_xlabel("Конверсий из 1000 юзеров")
ax.set_ylabel("плотность вероятности")
ax.legend(fontsize=8)

# --- Пуассона: заказы в час ---
ax = axes[0, 1]
ks = np.arange(orders_hour.min(), orders_hour.max() + 1)
ax.hist(orders_hour, bins=np.arange(ks[0] - 0.5, ks[-1] + 1.5, 1),
        density=True, color="#55A868", edgecolor="white", linewidth=0.4,
        label="выборка (2000 часов)")
ax.plot(ks, stats.poisson.pmf(ks, LAMBDA_HOUR), "o-", color="#C44E52",
        ms=3, lw=1, label=f"теория Poisson(λ={LAMBDA_HOUR})")
ax.set_title(f"Заказы в час на район: Poisson(λ={LAMBDA_HOUR})\n"
             f"шумовая полка ±√λ = ±{np.sqrt(LAMBDA_HOUR):.1f} заказов")
ax.set_xlabel("Заказов за час")
ax.legend(fontsize=8)

# --- нормальное: время доставки ---
ax = axes[1, 0]
ax.hist(delivery_min, bins=60, density=True, color="#8172B2",
        edgecolor="white", linewidth=0.3, label="выборка (3000 доставок)")
grid = np.linspace(delivery_min.min(), delivery_min.max(), 400)
ax.plot(grid, stats.norm.pdf(grid, MU_T, SIGMA_T), color="#C44E52", lw=2,
        label=f"теория N({MU_T:.0f}; {SIGMA_T:.0f})")
ax.set_title("Время доставки: Normal(34; 6)\n"
             "аддитивный процесс → симметрия, 68/95/99.7 в ±1/2/3σ")
ax.set_xlabel("Минут")
ax.legend(fontsize=8)

# --- логнормальное: чек ---
ax = axes[1, 1]
logbins = np.logspace(np.log10(check.min()), np.log10(check.max()), 70)
ax.hist(check, bins=logbins, density=True, color="#CCB974",
        edgecolor="white", linewidth=0.3, label="выборка (5000 заказов)")
grid = np.logspace(np.log10(check.min()), np.log10(check.max()), 400)
ax.plot(grid, stats.lognorm.pdf(grid, SIGMA_CHECK, scale=MEDIAN_CHECK),
        color="#C44E52", lw=2, label="теория Lognormal(ln 900; 0.55)")
ax.set_xscale("log")
ax.set_title("Чек: Lognormal(median=900; σ=0.55), ось x — log\n"
             "мультипликативный процесс → тяжёлый правый хвост")
ax.set_xlabel("Чек, ₽ (log-шкала)")
ax.legend(fontsize=8)

fig.suptitle("Зоопарк распределений «ЕдаДома»: одна ось — симуляция, кривая — теория",
             y=1.01)
fig.tight_layout()
fig.savefig(HERE / "practice_1_2_distributions.png", bbox_inches="tight")
plt.close(fig)

rows = [
    ["Binomial(1000; 0.12), mean", 1000 * P_CONV, conversions.mean(),
     np.sqrt(1000 * P_CONV * (1 - P_CONV)), conversions.std(ddof=1)],
    [f"Poisson({LAMBDA_HOUR}), mean", LAMBDA_HOUR, orders_hour.mean(),
     np.sqrt(LAMBDA_HOUR), orders_hour.std(ddof=1)],
    [f"Normal({MU_T:.0f}; {SIGMA_T:.0f}), mean", MU_T, delivery_min.mean(),
     SIGMA_T, delivery_min.std(ddof=1)],
    ["Lognormal(900; 0.55), median", MEDIAN_CHECK, np.median(check),
     np.nan, np.nan],
    ["Lognormal mean = e^(μ+σ²/2)", MEDIAN_CHECK * np.exp(SIGMA_CHECK ** 2 / 2),
     check.mean(), np.nan, np.nan],
]
print("=== Теория vs выборка (mean, sd) ===")
print(pd.DataFrame(rows, columns=["распределение", "теория", "выборка",
                                  "sd теория", "sd выборка"]).round(3).to_string(index=False))
print("Сохранено: practice_1_2_distributions.png")

# %% [markdown]
# ## 2. Сколько бинов в гистограмме: Стерджесс / Скотт / Фридман–Диаконис
#
# - Стерджесс: $k = 1 + \log_2 n$ (просто, заточено под нормальность, грубоват);
# - Скотт: $h = 3.5\,\sigma\, n^{-1/3}$ (ширина бина через σ);
# - Фридман–Диаконис: $h = 2\,IQR\, n^{-1/3}$ (IQR устойчив к хвостам).
#
# Считаем все три правила на сыром чеке, потом — на логарифмированном.

# %%
def bin_rules(x):
    """Возвращает словарь: число бинов по трём правилам."""
    x = np.asarray(x)
    n = x.size
    rng_x = x.max() - x.min()
    sturges = int(np.ceil(1 + np.log2(n)))
    h_scott = 3.5 * x.std(ddof=1) / n ** (1 / 3)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h_fd = 2 * iqr / n ** (1 / 3)
    return {
        "Стерджесс k = 1+log2(n)": sturges,
        "Скотт: h = 3.5σ/n^(1/3)": int(np.ceil(rng_x / h_scott)),
        "ФД: h = 2·IQR/n^(1/3)": int(np.ceil(rng_x / h_fd)),
    }

raw_rules = bin_rules(check)
log_rules = bin_rules(np.log(check))

print(f"=== Чек, n = {check.size}, диапазон {check.min():.0f}–{check.max():.0f} ₽ ===")
for name, k in raw_rules.items():
    print(f"сырые данные, {name:<28} → k = {k}")
print()
print("=== Тот же чек после log-преобразования ===")
for name, k in log_rules.items():
    print(f"log-данные,  {name:<28} → k = {k}")
print()
print("Вывод: на сырых скошенных деньгах правила расходятся в разы — Стерджесс")
print("грубит (мало бинов, тело замыливается), Скотт/ФД плодят бины в пустом хвосте.")
print("После log правила сходятся и картинка стабильна: сначала log, потом бины.")
print("(Чем тяжелее хвост, тем сильнее разрыв: на выручке из практики 1.1, σ=1.4,")
print(" Скотт и ФД дают уже сотни бинов в линейной шкале.)")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (name, k) in zip(axes, raw_rules.items()):
    ax.hist(check, bins=k, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_title(f"{name}\nk = {k} бинов", fontsize=10)
    ax.set_xlabel("Чек, ₽")
    ax.set_ylabel("число заказов (log)")
fig.suptitle("Одни данные — три правила бинов: форму гистограммы выбираете вы", y=1.05)
fig.tight_layout()
fig.savefig(HERE / "practice_1_2_bins.png", bbox_inches="tight")
plt.close(fig)
print("Сохранено: practice_1_2_bins.png")

# %% [markdown]
# ## 3. Число заказов на юзера: MoM vs MLE + ловушка overdispersion
#
# **Метод моментов (MoM):** приравниваем теоретическое среднее к выборочному:
# $E[X] = \lambda = \bar{x}$.
#
# **MLE:** ищем максимум log-правдоподобия численно — не верь формуле, проверь:
# $\ell(\lambda) = \sum_i \ln P(X = x_i; \lambda) \to \max_\lambda$.
#
# Потом то же на сверхдисперсных юзерах (отрицательное биномиальное с той же
# средней): Пуассон, подобранный по среднему, систематически «уже» реальности.

# %%
orders_user = rng.poisson(2.2, size=1000)          # пуассоновские юзеры
lam_mom = orders_user.mean()                        # метод моментов

def neg_loglik(lam):                                # −ℓ(λ) для минимизации
    return -stats.poisson.logpmf(orders_user, lam).sum()

res = minimize_scalar(neg_loglik, bounds=(0.1, 10.0), method="bounded")
lam_mle = res.x

print("=== Пуассон для числа заказов на юзера (n = 1000, истинная λ = 2.2) ===")
print(f"MoM:  λ̂ = выборочное среднее = {lam_mom:.4f}")
print(f"MLE:  λ̂ = argmax ℓ(λ)        = {lam_mle:.4f}  (совпали, как и обещает теория)")
print(f"Проверка пуассоновости: mean = {orders_user.mean():.2f}, "
      f"var = {orders_user.var(ddof=1):.2f}, var/mean = "
      f"{orders_user.var(ddof=1) / orders_user.mean():.2f} (у Пуассона = 1)")

# --- сверхдисперсные юзеры: NegBinomial с той же средней 2.2 и var ≈ 6.6 ---
R_NB = 1.1                                          # параметр «разброса λ по юзерам»
P_NB = R_NB / (R_NB + 2.2)
orders_nb = rng.negative_binomial(R_NB, P_NB, size=1000)
lam_nb = orders_nb.mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, data, title in [
    (axes[0], orders_user, "Пуассоновские юзеры: подгонка честная"),
    (axes[1], orders_nb, "Сверхдисперсные юзеры: Пуассон «уже» реальности"),
]:
    ks = np.arange(0, max(data.max(), 12) + 1)
    ax.hist(data, bins=np.arange(-0.5, ks[-1] + 1.5, 1), density=True,
            color="#55A868", edgecolor="white", linewidth=0.4, label="выборка")
    ax.plot(ks, stats.poisson.pmf(ks, data.mean()), "o-", color="#C44E52",
            ms=4, lw=1.2, label=f"Пуассон(λ̂ = x̄ = {data.mean():.2f})")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Заказов на юзера")
    ax.legend(fontsize=8)
axes[0].set_ylabel("доля юзеров")
fig.suptitle("Подгонка Пуассона по среднему: когда var/mean ≈ 1 — ок, когда ≫ 1 — нет",
             y=1.04)
fig.tight_layout()
fig.savefig(HERE / "practice_1_2_poisson_fit.png", bbox_inches="tight")
plt.close(fig)

print()
print("=== Сверхдисперсные юзеры (отрицательное биномиальное, mean тот же) ===")
print(f"mean = {orders_nb.mean():.2f}, var = {orders_nb.var(ddof=1):.2f}, "
      f"var/mean = {orders_nb.var(ddof=1) / orders_nb.mean():.2f}")
print(f"Пуассон с λ̂ = {lam_nb:.2f} предсказывает sd = √λ̂ = {np.sqrt(lam_nb):.2f}, "
      f"реальный sd = {orders_nb.std(ddof=1):.2f}")
print("Вывод: поюзерные счётчики сверхдисперсны (var/mean ≫ 1) — берите")
print("отрицательное биномиальное или бутстреп, а не Пуассон.")
print("Сохранено: practice_1_2_poisson_fit.png")

# %% [markdown]
# ## 4. Чек: MoM vs MLE для логнормального распределения
#
# - **MoM** решает систему по $\bar{x}$ и $s^2$ сырых данных:
#   $\hat\sigma^2 = \ln(1 + s^2/\bar{x}^2)$, $\hat\mu = \ln\bar{x} - \hat\sigma^2/2$;
# - **MLE** работает в логарифмах: $\hat\mu = \overline{\ln x}$, $\hat\sigma = sd(\ln x)$.
#
# Для Пуассона методы совпали, здесь — нет: моменты сырых денег тащат хвостовые киты.

# %%
xbar, s = check.mean(), check.std(ddof=1)
sigma2_mom = np.log(1 + (s / xbar) ** 2)
mu_mom = np.log(xbar) - sigma2_mom / 2
mu_mle, sigma_mle = np.log(check).mean(), np.log(check).std(ddof=0)

cmp = pd.DataFrame({
    "истина": [np.log(MEDIAN_CHECK), SIGMA_CHECK,
               MEDIAN_CHECK, MEDIAN_CHECK * np.exp(SIGMA_CHECK ** 2 / 2)],
    "MoM (сырые моменты)": [mu_mom, np.sqrt(sigma2_mom),
                            np.exp(mu_mom), np.exp(mu_mom + sigma2_mom / 2)],
    "MLE (по логарифмам)": [mu_mle, sigma_mle,
                            np.exp(mu_mle), np.exp(mu_mle + sigma_mle ** 2 / 2)],
}, index=["μ", "σ", "implied median, ₽", "implied mean, ₽"])

print("=== Логнормальный чек: MoM vs MLE (n = 5000, чистые данные) ===")
print(cmp.round(3).to_string())
print(f"Выборочные: median = {np.median(check):.0f} ₽, mean = {xbar:.0f} ₽")
print("На чистых данных при большом n методы близки. А теперь киты.")

# --- стресс-тест: подмешиваем 2% корпоративных заказов (медиана 15 000 ₽) ---
whales = rng.lognormal(mean=np.log(15000.0), sigma=0.6, size=100)
mix = np.concatenate([check, whales])

s2 = np.log(1 + (mix.std(ddof=1) / mix.mean()) ** 2)
mu_mom_m = np.log(mix.mean()) - s2 / 2
sigma_mom_m = np.sqrt(s2)
mu_mle_m, sigma_mle_m = np.log(mix).mean(), np.log(mix).std(ddof=0)

print()
print("=== Тот же чек + 2% китов (корпоративные заказы, медиана 15 000 ₽) ===")
print(f"MoM: σ̂ = {sigma_mom_m:.3f}, implied median = {np.exp(mu_mom_m):.0f} ₽, "
      f"implied mean = {np.exp(mu_mom_m + sigma_mom_m ** 2 / 2):.0f} ₽")
print(f"MLE: σ̂ = {sigma_mle_m:.3f}, implied median = {np.exp(mu_mle_m):.0f} ₽, "
      f"implied mean = {np.exp(mu_mle_m + sigma_mle_m ** 2 / 2):.0f} ₽")
print(f"Выборочные: median = {np.median(mix):.0f} ₽, mean = {mix.mean():.0f} ₽")
print()
print(f"Вывод: 2% китов разнесли MoM (σ̂ {np.sqrt(sigma2_mom):.2f} → "
      f"{sigma_mom_m:.2f}, «медиана» уехала к {np.exp(mu_mom_m):.0f} ₽),")
print(f"MLE по логарифмам почти не двинулся (σ̂ {sigma_mle:.2f} → "
      f"{sigma_mle_m:.2f}, «медиана» {np.exp(mu_mle):.0f} → "
      f"{np.exp(mu_mle_m):.0f} ₽ при выборочной {np.median(mix):.0f} ₽).")
print("Моменты сырых денег хрупки, логарифмы — прочные.")
print("Для тяжелохвостых метрик MLE — рабочий дефолт.")

# %% [markdown]
# ## 5. ECDF vs гистограмма на времени доставки
#
# Гистограмме нужны бины; эмпирической функции распределения (ECDF) — нет:
# $\hat F(x)$ = доля наблюдений ≤ x. Читаем: медиана (0.5), P90 (0.9),
# доля доставок быстрее 40 минут (вертикаль при x = 40).

# %%
x_sorted = np.sort(delivery_min)
ecdf_y = np.arange(1, x_sorted.size + 1) / x_sorted.size

med = np.median(delivery_min)
p90 = np.percentile(delivery_min, 90)
f40 = (delivery_min <= 40).mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

axes[0].hist(delivery_min, bins=40, density=True, color="#8172B2",
             edgecolor="white", linewidth=0.3)
grid = np.linspace(delivery_min.min(), delivery_min.max(), 300)
axes[0].plot(grid, stats.norm.pdf(grid, MU_T, SIGMA_T), color="#C44E52", lw=2)
axes[0].set_title("Гистограмма: форма — но нужны бины")
axes[0].set_xlabel("Время доставки, мин")
axes[0].set_ylabel("плотность")

axes[1].step(x_sorted, ecdf_y, where="post", color="#4C72B0", lw=1.6,
             label="ECDF — без бинов и параметров")
axes[1].axhline(0.5, color="gray", ls="--", lw=1)
axes[1].axhline(0.9, color="gray", ls=":", lw=1)
axes[1].axvline(40, color="#C44E52", lw=1.5)
axes[1].plot([med], [0.5], "o", color="#C44E52")
axes[1].plot([p90], [0.9], "o", color="#C44E52")
axes[1].set_title("ECDF: читаем квантили и доли напрямую")
axes[1].set_xlabel("Время доставки, мин")
axes[1].set_ylabel("доля доставок ≤ x")
axes[1].legend(fontsize=9, loc="lower right")

fig.suptitle(f"Время доставки: медиана = {med:.1f} мин, P90 = {p90:.1f} мин, "
             f"быстрее 40 мин — {f40 * 100:.0f}% доставок", y=1.03)
fig.tight_layout()
fig.savefig(HERE / "practice_1_2_ecdf.png", bbox_inches="tight")
plt.close(fig)

print("=== Чтение ECDF (время доставки) ===")
print(f"медиана (F=0.5):                 {med:.1f} мин")
print(f"P90 (F=0.9):                     {p90:.1f} мин  ← язык SLA")
print(f"доля доставок быстрее 40 мин:    {f40 * 100:.1f}%  → доля опоздавших "
      f"{(1 - f40) * 100:.1f}%")
print(f"проверка норм.приближения: P(X≤40) по N(34;6) = "
      f"{stats.norm.cdf(40, MU_T, SIGMA_T) * 100:.1f}%")
print("Сохранено: practice_1_2_ecdf.png")

# %% [markdown]
# ## 6. QQ-plot: логнормальность, увиденная глазами
#
# Слева — QQ-plot сырого чека против нормального распределения: точки
# устойчиво выше прямой в правом хвосте → хвост тяжелее нормального.
# Справа — QQ-plot ln(чека): почти прямая линия → чек логнормален.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

stats.probplot(check, dist="norm", plot=axes[0])
axes[0].set_title("QQ-plot: сырой чек vs Normal\nизгиб вверх справа — хвост тяжелее",
                  fontsize=10)
axes[0].set_xlabel("Теоретические квантили N(0;1)")
axes[0].set_ylabel("Выборочные квантили чека, ₽")

stats.probplot(np.log(check), dist="norm", plot=axes[1])
axes[1].set_title("QQ-plot: ln(чек) vs Normal\nпочти прямая — данные логнормальны",
                  fontsize=10)
axes[1].set_xlabel("Теоретические квантили N(0;1)")
axes[1].set_ylabel("Выборочные квантили ln(чек)")

fig.tight_layout()
fig.savefig(HERE / "practice_1_2_qq.png", bbox_inches="tight")
plt.close(fig)

r_raw = stats.probplot(check, dist="norm")[1][2]
r_log = stats.probplot(np.log(check), dist="norm")[1][2]
print("=== QQ-plot чека ===")
print(f"коэффициент корреляции точек QQ (сырой чек):  R² = {r_raw:.3f}")
print(f"коэффициент корреляции точек QQ (ln чек):      R² = {r_log:.3f}")
print("Вывод: деньги до log-преобразования не нормальны (хвост уходит вверх),")
print("после — ложатся на прямую. Это и есть диагноз «логнормальный чек».")
print("Сохранено: practice_1_2_qq.png")

# %% [markdown]
# ## Итог практики 1.2
#
# - У каждой метрики своё естественное распределение: конверсия — Binomial,
#   события — Poisson, время — Normal, деньги — Lognormal.
# - Формулы mean/sd по параметрам совпали с выборкой — их можно брать в работу
#   (шумовые полки для алертов).
# - Число бинов меняет историю: Стерджесс/Скотт/ФД + log-шкала для денег.
# - MoM и MLE для Пуассона совпадают (λ̂ = x̄); для логнормалы MLE точнее.
# - Поюзерные счётчики сверхдисперсны — Пуассон для них врёт по разбросу.
# - ECDF и QQ-plot — бесплатные диагносты: без бинов, без веры на слово.

# %%
print("================ ГЛАВНЫЕ ЧИСЛА УРОКА 1.2 ================")
print(f"Конверсия Binom(1000; 0.12): sd = ±{np.sqrt(1000 * P_CONV * (1 - P_CONV)):.1f} "
      f"конверсий (≈ ±{np.sqrt(P_CONV * (1 - P_CONV) / 1000) * 100:.2f} п.п.)")
print(f"Заказы Poisson(40): шум ±√40 = ±{np.sqrt(LAMBDA_HOUR):.1f} заказов/час")
print(f"Чек: mean/median = {check.mean() / np.median(check):.2f}, "
      f"теория e^(σ²/2) = {np.exp(SIGMA_CHECK ** 2 / 2):.2f} при σ = {SIGMA_CHECK}")
print(f"Бины на чеке (n=5000): Стерджесс {raw_rules['Стерджесс k = 1+log2(n)']}, "
      f"Скотт {raw_rules['Скотт: h = 3.5σ/n^(1/3)']}, "
      f"ФД {raw_rules['ФД: h = 2·IQR/n^(1/3)']} (сырые данные)")
print(f"Пуассон: MoM λ̂ = {lam_mom:.3f} = MLE λ̂ = {lam_mle:.3f}; "
      f"overdispersion: var/mean = {orders_nb.var(ddof=1) / orders_nb.mean():.1f}")
print(f"Логнормала: MLE σ̂ = {sigma_mle:.3f} (истина {SIGMA_CHECK}), "
      f"MoM σ̂ = {np.sqrt(sigma2_mom):.3f}; с 2% китов: MoM σ̂ = {sigma_mom_m:.3f} "
      f"vs MLE σ̂ = {sigma_mle_m:.3f}")
print(f"ECDF доставки: медиана {med:.0f} мин, P90 {p90:.0f} мин, "
      f"≤40 мин: {f40 * 100:.0f}%")
print(f"QQ-plot R²: сырой чек {r_raw:.3f} vs ln(чек) {r_log:.3f}")
print("PNG: practice_1_2_distributions.png, practice_1_2_bins.png,")
print("     practice_1_2_poisson_fit.png, practice_1_2_ecdf.png, practice_1_2_qq.png")
print("========================================================")
