# %% [markdown]
# # Практика 1.1 — Описательные статистики: выручка «ЕдаДома» и её киты
#
# Сквозной кейс: сервис доставки еды «ЕдаДома». Метрика — месячная выручка
# активных юзеров (₽). Она ведёт себя как деньги во всех продуктах:
# горб слева, тяжёлый правый хвост, киты.
#
# План:
# 1. Генерируем логнормальную выручку 5 000 юзеров (seed фиксируем).
# 2. Считаем описательные статистики: mean / median / std / квантили / skew.
# 3. Удаляем топ-1% юзеров и смотрим, что стало со средним и медианой.
# 4. Графики: гистограмма (линейная и log-шкала), боксплот «до/после».
# 5. Закон малых чисел: разброс среднего в группах по 10 / 50 / 500 юзеров.
# 6. Сводная таблица по городам: mean + median + count рядом.
#
# Культура курса: не верь одному числу — проверь графиком и симуляцией.

# %%
import matplotlib

matplotlib.use("Agg")  # сохраняем png без экрана
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["figure.dpi"] = 110
HERE = Path(__file__).resolve().parent  # png лягут рядом со скриптом

# Все случайности — из одного генератора с фиксированным seed:
# результат воспроизводим от запуска к запуску.
rng = np.random.default_rng(2026)

# %% [markdown]
# ## 1. Данные: логнормальная выручка 5 000 юзеров
#
# Модель: ln(выручка) ~ N(mu, sigma^2). Медиана = e^mu = 900 ₽,
# тяжёлый хвост задаёт sigma = 1.4. Почему деньги именно такие — урок 1.2.

# %%
N = 5000
MU = np.log(900.0)   # медиана ~ 900 ₽/мес
SIGMA = 1.4          # тяжесть хвоста

revenue = rng.lognormal(mean=MU, sigma=SIGMA, size=N)
df = pd.DataFrame({"user_id": [f"u{i:05d}" for i in range(N)],
                   "revenue": np.round(revenue, 2)})

df.head(10)

# %% [markdown]
# ## 2. Описательные статистики
#
# Смотрим полный набор: `describe()` + skewness. Ключевой вопрос урока —
# насколько среднее оторвано от медианы.

# %%
desc = df["revenue"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
skew = stats.skew(df["revenue"])
cv = df["revenue"].std() / df["revenue"].mean()
iqr = df["revenue"].quantile(0.75) - df["revenue"].quantile(0.25)
tukey_hi = df["revenue"].quantile(0.75) + 1.5 * iqr

print("=== Описательные статистики выручки (n = %d) ===" % N)
print(desc.round(1).to_string())
print(f"skewness (скошенность):      {skew:.2f}")
print(f"IQR = P75 - P25:             {iqr:.0f} ₽")
print(f"Верхний забор Тьюки:         {tukey_hi:.0f} ₽ "
      f"(выше него выбросы по правилу 1.5*IQR)")
print(f"CV (std/mean):               {cv:.2f}")
print(f"mean / median:               {df['revenue'].mean() / df['revenue'].median():.2f}")
print(f"Доля юзеров выше забора:     {(df['revenue'] > tukey_hi).mean() * 100:.1f}%")

# %% [markdown]
# Среднее почти втрое выше медианы, skew > 5: классическая картина денег.
# Забор Тьюки формально «выбрасывает» заметную долю юзеров — это не приказ
# на удаление, а сигнал посмотреть на хвост.

# %% [markdown]
# ## 3. Убираем топ-1% юзеров: кто держит среднее?

# %%
p99 = df["revenue"].quantile(0.99)
top1_mask = df["revenue"] >= p99
df_trim = df[~top1_mask]

before = df["revenue"]
after = df_trim["revenue"]

comparison = pd.DataFrame({
    "все юзеры (n=5000)":  [before.mean(), before.median(), before.std(),
                            before.quantile(0.90), before.max()],
    "без топ-1% (n=4950)": [after.mean(), after.median(), after.std(),
                            after.quantile(0.90), after.max()],
}, index=["mean, ₽", "median, ₽", "std, ₽", "P90, ₽", "max, ₽"])

top1_share = before[top1_mask].sum() / before.sum()

print(f"=== До / после удаления топ-1% (порог = P99 = {p99:.0f} ₽) ===")
print(comparison.round(0).to_string())
print()
print(f"Доля выручки топ-1% юзеров:        {top1_share * 100:.1f}%")
print(f"Среднее упало на:                  {(1 - after.mean() / before.mean()) * 100:.1f}%")
print(f"Медиана упала на:                  {(1 - after.median() / before.median()) * 100:.1f}%")
print(f"STD упал на:                       {(1 - after.std() / before.std()) * 100:.1f}%")
print("Вывод: среднее живёт в хвосте, медиана — в середине распределения.")

# %% [markdown]
# ## 4. График 1: гистограмма в линейной шкале vs log-шкале
#
# Одна и та же данных, две шкалы. В линейной шкале логнормала —
# «одна палка и пустота»: хвост нечитаем. В log-шкале проявляется форма.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].hist(before, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.3)
axes[0].set_title("Линейная шкала: одна палка и хвост-невидимка")
axes[0].set_xlabel("Выручка, ₽")
axes[0].set_ylabel("Число юзеров")
axes[0].axvline(before.mean(), color="red", lw=2,
                label=f"mean = {before.mean():.0f} ₽")
axes[0].axvline(before.median(), color="green", lw=2,
                label=f"median = {before.median():.0f} ₽")
axes[0].legend()

axes[1].hist(before, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.3)
axes[1].set_xscale("log")
axes[1].set_title("Log-шкала: видно форму распределения")
axes[1].set_xlabel("Выручка, ₽ (log)")
axes[1].axvline(before.mean(), color="red", lw=2, label="mean")
axes[1].axvline(before.median(), color="green", lw=2, label="median")
axes[1].legend()

fig.suptitle("Месячная выручка юзеров «ЕдаДома»: линейная vs log-шкала", y=1.03)
fig.tight_layout()
fig.savefig(HERE / "practice_1_1_hist_logscale.png", bbox_inches="tight")
plt.close(fig)
print("Сохранено: practice_1_1_hist_logscale.png")

# %% [markdown]
# ## 5. График 2: боксплот «до/после» + забор Тьюки
#
# Усы боксплота заканчиваются на 1.5*IQR от квартилей; точки снаружи —
# кандидаты в выбросы. Слева — рой китов, справа — что стало после среза топ-1%.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

axes[0].boxplot(before, vert=False, showfliers=True,
                flierprops=dict(marker=".", markersize=3, alpha=0.4))
axes[0].set_title(f"Все юзеры: киты видны как рой точек\n(доля выше забора: "
                  f"{(before > tukey_hi).mean() * 100:.1f}%)")
axes[0].set_xlabel("Выручка, ₽")

axes[1].boxplot(after, vert=False, showfliers=True,
                flierprops=dict(marker=".", markersize=3, alpha=0.4))
axes[1].set_title("Без топ-1%: коробка почти не изменилась")
axes[1].set_xlabel("Выручка, ₽")

fig.suptitle(f"Боксплот выручки. Забор Тьюки (слева): {tukey_hi:.0f} ₽", y=1.05)
fig.tight_layout()
fig.savefig(HERE / "practice_1_1_boxplot.png", bbox_inches="tight")
plt.close(fig)
print("Сохранено: practice_1_1_boxplot.png")

# %% [markdown]
# Коробка (Q1–Q3, медиана) почти не сдвинулась — медианные статистики
# устойчивы к хвосту. Изменились именно «усадка» по x и рой выбросов.
# Обратим внимание: топ-1% юзеров — это ~50 человек из 5 000,
# но без них график становится «приличным». Не путать с «правильным».

# %% [markdown]
# ## 6. Закон малых чисел: разброс среднего в малых группах
#
# 2 000 раз сэмплируем группы по 10 / 50 / 500 юзеров и считаем среднюю
# выручку каждой группы. Сравниваем разброс с теоретическим $\sigma/\sqrt{n}$.
# Это и есть «шумовая полка»: любое изменение метрики на сегменте надо
# сравнивать с ней, а не с нулём.

# %%
REPS = 2000
group_sizes = [10, 50, 500]
true_std = before.std(ddof=1)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=False)
print("=== Закон малых чисел: разброс среднего по группам ===")
for ax, g in zip(axes, group_sizes):
    means = np.array([rng.choice(before.values, size=g, replace=False).mean()
                      for _ in range(REPS)])
    sd_emp = means.std(ddof=1)
    sd_theory = true_std / np.sqrt(g)
    ax.hist(means, bins=60, color="#55A868", edgecolor="white", linewidth=0.3)
    ax.axvline(before.mean(), color="black", lw=1.5, label="истинное среднее")
    ax.set_title(f"n = {g} в группе\nSD средних: {sd_emp:.0f} ₽ "
                 f"(теория σ/√n = {sd_theory:.0f})")
    ax.set_xlabel("Средняя выручка группы, ₽")
    ax.legend(fontsize=8)
    print(f"n = {g:>3}: SD средних по группам = {sd_emp:7.0f} ₽ "
          f"(теория {sd_theory:7.0f}); min-max средних: "
          f"{means.min():.0f} … {means.max():.0f} ₽")

fig.suptitle(f"Средние групп мечутся: шум падает как 1/√n (SD выручки = {true_std:.0f} ₽)",
             y=1.04)
fig.tight_layout()
fig.savefig(HERE / "practice_1_1_small_numbers.png", bbox_inches="tight")
plt.close(fig)
print("Сохранено: practice_1_1_small_numbers.png")

# %% [markdown]
# ## 7. Сводная таблица по городам
#
# Правильная сводная: рядом mean, median и **count**. Самый маленький
# город даст самый «эффектный» результат — потому что он самый шумный.

# %%
cities = rng.choice(["Москва", "Казань", "Сочи"], size=N, p=[0.5, 0.35, 0.15])
df["city"] = cities

pivot = df.pivot_table(index="city", values="revenue",
                       aggfunc=["count", "mean", "median", "std"])
pivot.columns = ["n", "mean, ₽", "median, ₽", "std, ₽"]

# Шумовая полка среднего каждого города: sigma/sqrt(n)
pivot["шум mean (σ/√n), ₽"] = pivot["std, ₽"] / np.sqrt(pivot["n"])
pivot["mean/median"] = pivot["mean, ₽"] / pivot["median, ₽"]
pivot = pivot.round(0).sort_values("n")

print("=== Сводная по городам ===")
print(pivot.to_string())
print()
noisiest = pivot.index[0]
print(f"Самый маленький город: {noisiest} (n = {int(pivot.loc[noisiest, 'n'])}). "
      f"Его «особенности» легко объясняются шумом ±{pivot.loc[noisiest, 'шум mean (σ/√n), ₽']:.0f} ₽ "
      f"на одно стандартное отклонение.")
print("Вывод: рядом с каждым числом в сводной обязан стоять n; "
      "сегмент без n — не сегмент, а лотерея.")

# %% [markdown]
# ## Итог практики 1.1
#
# - Деньги (выручка, чек, LTV) скошены вправо: mean >> median, киты тянут среднее.
# - Топ-1% юзеров держит двузначную долю выручки; их удаление рушит mean
#   и почти не трогает median.
# - Смотреть деньги нужно в log-шкале и боксплотом, а не одним числом.
# - Шум оценок падает как σ/√n: на группах в десятки наблюдений «эффекты» —
#   это чаще всего закон малых чисел, а не продукт.

# %%
print("================ ГЛАВНЫЕ ЧИСЛА УРОКА 1.1 ================")
print(f"n = {N}, медиана = {before.median():.0f} ₽, среднее = {before.mean():.0f} ₽ "
      f"(mean/median = {before.mean() / before.median():.2f})")
print(f"skewness = {skew:.1f}  →  сильная правая скошенность")
print(f"Топ-1% юзеров (>= {p99:.0f} ₽) держат {top1_share * 100:.1f}% выручки")
print(f"Без топ-1%: mean падает на {(1 - after.mean() / before.mean()) * 100:.1f}%, "
      f"median — на {(1 - after.median() / before.median()) * 100:.1f}%")
print(f"Забор Тьюки = {tukey_hi:.0f} ₽; выше него {(before > tukey_hi).mean() * 100:.1f}% юзеров")
print(f"Шум среднего: n=10 → ±{true_std / np.sqrt(10):.0f} ₽; "
      f"n=500 → ±{true_std / np.sqrt(500):.0f} ₽ (одно SD)")
print("PNG: practice_1_1_hist_logscale.png, practice_1_1_boxplot.png, "
      "practice_1_1_small_numbers.png")
print("========================================================")
