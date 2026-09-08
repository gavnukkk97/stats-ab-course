---
type: source
kind: paper
module: [М3, М4]
tags: [аб, cuped, чувствительность, paper]
---
# Deng, Xu, Kohavi, Walker — CUPED (WSDM 2013)

Deng, Xu, Kohavi, Walker «Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data» (WSDM 2013) — оригинальная статья CUPED. Классика, на которой стоит всё современное снижение дисперсии.

Чем ценен: первоисточник. CUPED = Control variates Utilizing Pre-Experiment Data: Y_cv = Y − θ·X, θ = cov(Y,X)/var(X); X = значение той же метрики в pre-period → не зависит от тритмана, Var падает ≈ в (1 − ρ²). Результаты на Bing: −40–50% дисперсии ≈ удвоение трафика. Ковариата не обязана быть той же метрикой; эквивалент ANCOVA/regression adjustment; pre-period ≥ 2 недель; эффекты новизны исключаются сравнением внутри тех же юзеров.

Закрывает: [[CUPED]], [[ML variance reduction]], [[Мощность]].

Оригинал: https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf ; конспект — `Материалы/research-sensitivity.md` §1.

Модули: [[М3 Статистика для АБ]] (уроки 3.4–3.5).
