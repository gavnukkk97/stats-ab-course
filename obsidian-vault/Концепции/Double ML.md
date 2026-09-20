---
type: concept
module: М6
tags: [каузальность, ml]
---
# Double ML

Double/debiased machine learning использует ортогональные score-функции и cross-fitting, чтобы уменьшить чувствительность оценки к ошибкам nuisance-моделей. Это способ оценки при заданной идентификации, а не лечение скрытого confounding.

В частично линейной модели Y=θD+g(X)+ε оценивают m(X)=E[D|X] и ℓ(X)=E[Y|X] на других folds и регрессируют Y−ℓ(X) на D−m(X). При бинарном D и неоднородном τ(X) предел такого PLR-коэффициента равен E[e(X)(1−e(X))τ(X)]/E[e(X)(1−e(X))], а не обычному ATE. Для ATE выбирают соответствующий score, например cross-fitted AIPW в interactive regression model.

Ортогональность и cross-fitting не обещают устранить любое смещение. Нужны идентификационные допущения, overlap, достаточная точность nuisance-функций и условия для асимптотического интервала.

[practice_6_4.py](../../course/modules/M6/practice/practice_6_4.py) сравнивает PLR и AIPW на данных с неоднородным эффектом. [[Doubly robust]], [[Propensity score]], [[ATE]].

Модуль: [[М6 Каузальный вывод]]

## Где почитать
- [[koch-kir — Causal Inference]] (§Double ML)
- [[Ci2Lab — Applied Causal Inference]] README (Chernozhukov et al. в рекомендованных книгах)
- [[shelter_analytics]] «AI в Causal Inference»
