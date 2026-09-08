---
type: source
kind: paper
module: [М5, М4]
tags: [байес, expected-loss, paper]
---
# Stucchio — Easy Evaluation of Decision Rules

Chris Stucchio «Easy Evaluation of Decision Rules in Bayesian A/B testing» (блог, 2014) + VWO SmartStats Technical Whitepaper (2015).

Чем ценен: источник правила остановки по expected loss: остановись, когда ожидаемые потери выбора худшего варианта < «threshold of caring» ε. Замкнутая форма через бета-апостериоры: E[max(x−y,0)] через функцию h(a,b,c,d) Миллера — O(1) без интегрирования. VWO whitepaper — продакшн-версия в бизнес-терминах (риск в единицах конверсии), аргументы против фиксированного горизонта t-теста. ОБЯЗАТЕЛЬНАЯ ОГОВОРКА: «peeking immunity» — миф (David Robinson, varianceexplained.org): error-рейты при подглядывании растут, контролируются ожидаемые потери; сочетать с [[mSPRT]] / always-valid. Код практикума: github.com/twitter/bayesian-ab-testing (Guo, Kannan, Long, Maples, Miratrix, Tang).

Закрывает: [[Expected loss]], [[Байесовский AB-тест]], [[Подглядывание]].

Оригинал: https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html ; конспект — `Материалы/research-gaps.md` §Задача 2 (пп. 4–5).

Модули: [[М5 Байесовские методы]] (урок 5.3), [[М4 АБ на практике]].
