---
type: concept
module: М6
tags: [каузальность, наблюдательные]
---
# IPW

Inverse Propensity Weighting — взвешивание наблюдений на обратные вероятности назначения: юниты, «маловероятные» для своей группы, получают больший вес, восстанавливая псевдопопуляцию без смещения отбора.

## Зачем
Один из базовых идентифицирующих оценок ATE на наблюдательных данных; мост от [[Propensity score|propensity score]] к практическому оценивателю.

## Ключевые идеи
- Происхождение: importance sampling в missing data-рамке Рубина (Deng гл.4)
- Чувствительность к экстремальным весам (маленькие propensity → огромные веса → взрыв дисперсии) — практическое ограничение; стабилизация/укорочение хвостов
- В AIPW/[[Doubly robust]]: IPW — один из двух компонентов двойной защиты
- В AI-агентах Netflix: AIPW + EconML как вычислительное ядро CI-агентов (shelter, разбор 2026)
- Синтаксис связи: weight → weighted mean → ATE; регрессионная альтернатива — контроль ковариат

Связано: [[Propensity score]], [[Doubly robust]], [[ATE]], [[Double ML]]

Модуль: [[М6 Каузальный вывод]]

## Где почитать
- [[Deng — Causal Inference and Its Applications]] гл.4
- [[koch-kir — Causal Inference]]
- [[shelter_analytics]] «AI в Causal Inference» (AIPW + EconML)
