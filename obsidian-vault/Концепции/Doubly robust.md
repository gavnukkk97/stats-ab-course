---
type: concept
module: М6
tags: [каузальность, наблюдательные]
---
# Doubly robust

AIPW объединяет outcome-модели μd(X) и propensity e(X):

ATE = E[μ1(X)−μ0(X) + D(Y−μ1(X))/e(X) − (1−D)(Y−μ0(X))/(1−e(X))].

При consistency, conditional exchangeability, overlap и регулярности оценка консистентна, если верна propensity-модель либо обе outcome-модели. При верной outcome-модели остаточные слагаемые имеют нулевое среднее. При верном propensity взвешенные остатки компенсируют ошибки outcome-моделей; их средние не обязаны быть нулевыми.

Две модели вполне могут ошибаться одновременно. DR не гарантирует меньшую дисперсию, лучший конечновыборочный результат или корректный CI при произвольном ML; для inference нужны дополнительные условия, часто cross-fitting.

[[IPW]], [[Double ML]], [[ATE]]. Практика: [practice_6_4.py](../../course/modules/M6/practice/practice_6_4.py), сценарии misspecification.

Модуль: [[М6 Каузальный вывод]]

## Где почитать
- [[koch-kir — Causal Inference]] (§doubly robust)
- [[Deng — Causal Inference and Its Applications]] гл.4, 10.2.3
- [[shelter_analytics]] «AI в Causal Inference» (AIPW + EconML)
