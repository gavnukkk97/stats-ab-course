---
type: concept
module: М6
tags: [каузальность, ml]
---
# Double ML

Double/Debiased ML (Chernozhukov): ML-модели предсказывают исход и лечение по ковариатам, затем каузальный эффект оценивается регрессией остатков на остатки (ортогонализация, теорема Фриша–Вауга–Ловелла).

## Зачем
Современный стандарт контроля многомерных ковариат в наблюдательных данных: ML выучивает нелинейные зависимости, ортогонализация убирает смещение регуляризации.

## Ключевые идеи
- Логика Фриш–Вауг–Ловелл: две ML-модели (outcome и treatment), кросс-фиттинг, регрессия остатков (koch-kir)
- Библиотеки: CausalML, DoWhy, EconML; рекомендованы и Ci2Lab (README) (koch-kir; Ci2Lab)
- Смежное: [[Doubly robust]]/AIPW — то же семейство «двойных/ортогональных» оценок (shelter: AIPW + EconML в CI-агентах Netflix)
- CVM-аналитика X5: кумулятивные эффекты и GCG — индустрийные кузены DML-подхода для офлайн-промо (abba_testing)
- Практика урока 6.4: DoWhy/EconML на uplift-кампании (план)

Связано: [[Doubly robust]], [[Propensity score]], [[IPW]], [[ATE]], [[CUPED]]

Модуль: [[М6 Каузальный вывод]]

## Где почитать
- [[koch-kir — Causal Inference]] (§Double ML)
- [[Ci2Lab — Applied Causal Inference]] README (Chernozhukov et al. в рекомендованных книгах)
- [[shelter_analytics]] «AI в Causal Inference»
