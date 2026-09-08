---
type: concept
module: М6
tags: [каузальность, наблюдательные]
---
# Propensity score

Вероятность получения лечения по ковариатам: e(X) = P(T=1|X). Обусловливание на score балансирует наблюдаемые ковариаты — как будто рандомизация (при unconfoundedness).

## Зачем
Сжимает многомерный контроль к одной переменной: matching/взвешивание/стратификация по score вместо матчинга по всем ковариатам.

## Ключевые идеи
- PSM в наборе квазиэкспериментов: Kohavi гл.11, HKU L7 (слайды + практика), обзор MCP (нужны богатые ковариаты)
- Трейд-офф жёсткого матчинга (дисперсия) vs мягкого (смещение) (koch-kir)
- Условие применимости: overlap/covariate overlap — лечения и контроля должны перекрываться по X (Deng гл.4)
- Связь с missing data и importance sampling: score = вероятность «наблюдаемости» (Deng гл.4)
- «Большой матчинг» (matchit в R): разные методы дают разные [[ATE]] — чувствительность к спецификации обязательна к отчёту (stats_for_science)
- Практика урока 6.4: PSM на датасете uplift-кампании, DoWhy/EconML (план)

Связано: [[IPW]], [[Doubly robust]], [[Конфаундер]], [[ATE]], [[Double ML]]

Модуль: [[М6 Каузальный вывод]]

## Где почитать
- [[koch-kir — Causal Inference]] (§conditioning)
- [[Kohavi — Trustworthy OCE]] гл.11; [[HKU — Digital Experimentation]] L7
- [[Deng — Causal Inference and Its Applications]] гл.4
- [[stats_for_science]] «Большой матчинг»
