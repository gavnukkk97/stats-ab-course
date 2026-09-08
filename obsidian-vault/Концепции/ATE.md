---
type: concept
module: М6
tags: [каузальность]
---
# ATE

Average Treatment Effect: E[Y(1) − Y(0)] — средний эффект лечения по всей популяции. Родственные: ATT (по лечёным), CATE (по подгруппам), ITE (индивидуальный, ненаблюдаем).

## Зачем
Целевая величина и A/B-теста, и каузальных методов: усреднённый ответ на «что будет, если показать всем». Рандомизация делает difference-in-means несмещённой оценкой ATE.

## Ключевые идеи
- HKU L2: treatment effects и оценка ATE — первый кадр статистики экспериментов; RCT как two-sample problem (t/z-тесты) для PATE и SATE (Deng гл.3, Imbens & Rubin)
- Ci2Lab CH-4: сквозной пример «пиво → счастье студентов Kronbar» — шаги 1–7 от допущений до ATE и контрфактов
- Гетерогенность: HTE/CATE по сегментам через OLS + interaction (HKU L5 Heterogenous.ipynb) — с осторожностью ([[Множественные сравнения]])
- В наблюдательных данных ATE оценивают: регрессия, [[Propensity score]], [[IPW]], [[Doubly robust]], [[Double ML]] — «Большой матчинг»: разные методы — разные ATE, чувствительность к спецификации (stats_for_science)
- Эффект в триггерной популяции ≠ общий ATE (HKU L5 triggering)

Связано: [[Potential outcomes]], [[Propensity score]], [[DiD]], [[t-тест]], [[Юнит рандомизации]]

Модуль: [[М6 Каузальный вывод]]

## Где почитать
- [[HKU — Digital Experimentation]] L2–L5
- [[Ci2Lab — Applied Causal Inference]] CH-4
- [[Deng — Causal Inference and Its Applications]] гл.3–4
- [[stats_for_science]] «Большой матчинг»
