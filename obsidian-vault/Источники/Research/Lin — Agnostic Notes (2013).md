---
type: source
kind: paper
module: [М2, М3]
tags: [аб, cuped, регрессия, paper]
---
# Lin — Agnostic Notes (2013)

Winston Lin «Agnostic Notes on Regression Adjustments to Experimental Data: Reexamining Freedman's Critique» (Annals of Applied Statistics 7(1), 2013) — ответ на критику Freedman (2008) про регрессионную поправку в RCT.

Чем ценен: теоретический фундамент «правильного CUPED». Оценщик: OLS Y на T, центрированные ковариаты X и полные взаимодействия T×X с гетероскедастичность-робастными SE. Результаты: (1) коэффициент при T состоятелен для ATE даже при misspecification; (2) асимптотическая дисперсия никогда не больше, чем у unadjusted difference-in-means — «adjustment ничего не стоит»; (3) без взаимодействий (кейс Freedman) гарантий нет → всегда включать взаимодействия и центрировать. ExP Deep Dive: CUPED с раздельными коэффициентами по рукам = ANCOVA2 = Lin-оценщик; ByteDance (arXiv 2509.13944) независимо рекомендует group-specific коэффициенты.

Закрывает: [[CUPED]], [[ML variance reduction]], [[ANOVA]].

Оригинал: doi 10.1214/12-AOAS584 (projecteuclid.org) ; конспект — `Материалы/research-sensitivity.md` раздел «CUPED — ANCOVA — regression adjustment».

Модули: [[М2 Проверка гипотез]], [[М3 Статистика для АБ]] (урок 3.5).
