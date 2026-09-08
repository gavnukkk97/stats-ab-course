---
type: source
kind: paper
module: [М3]
tags: [аб, cuped, ml, чувствительность, paper]
---
# Poyarkov et al — BDT adjustment (Yandex)

Poyarkov, Drutsa, Khalturin et al. (Yandex + Microsoft/Bing) «Boosted Decision Tree Regression Adjustment for Variance Reduction in Online Controlled Experiments» (KDD 2016).

Чем ценен: прямой предок CUPAC/ML-RATE — замена линейной regression adjustment на бустинг над решающими деревьями (регрессия Y на pre-experiment ковариаты): ловит нелинейности и взаимодействия. Валидация на 161 реальном онлайн-эксперименте (не симуляции!) — редкость для литературы. Итог: нелинейная поправка дополнительно снижает дисперсию относительно линейного CUPED на ряде метрик; встречаются метрики, где линейная версия почти не помогает, а BDT — да.

Закрывает: [[ML variance reduction]], [[CUPAC]], [[CUPED]].

Оригинал: https://kdd.org/kdd2016/papers/files/adf0653-poyarkovA.pdf ; DOI 10.1145/2939672.2939688 ; конспект — `Материалы/research-sensitivity.md` §5.

Модуль: [[М3 Статистика для АБ]] (урок 3.5).
