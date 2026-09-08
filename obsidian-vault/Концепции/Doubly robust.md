---
type: concept
module: М6
tags: [каузальность, наблюдательные]
---
# Doubly robust

Двойно робастные оценки (AIPW): комбинируют модель исхода (регрессия) и модель назначения ([[Propensity score|propensity]]/[[IPW]]); консистентны, если верна хотя бы одна из двух моделей.

## Зачем
Страховка от ошибки спецификации: обе модели сразу ошибаются редко — двойная защита делает оценку надёжнее каждого компонента по отдельности.

## Ключевые идеи
- Свойство «или-или» — формулируется у koch-kir и Deng (гл.10.2.3: связь с semiparametric efficiency)
- AIPW (augmented IPW) — практическая реализация; ядро пайплайнов CI-агентов Netflix (AIPW + EconML, placebo-тесты, валидация на ACIC 2016) (shelter)
- Родственная логика у [[Double ML]]: ML-модели для исхода и тритмента + регрессия остатков (Фриш–Вауг–Ловелл)
- Смежная тема М3: CUPED как regression adjustment — кузнец doubly robust идей в чувствительности экспериментов (Deng гл.10)

Связано: [[IPW]], [[Propensity score]], [[Double ML]], [[CUPED]], [[ATE]]

Модуль: [[М6 Каузальный вывод]]

## Где почитать
- [[koch-kir — Causal Inference]] (§doubly robust)
- [[Deng — Causal Inference and Its Applications]] гл.4, 10.2.3
- [[shelter_analytics]] «AI в Causal Inference» (AIPW + EconML)
