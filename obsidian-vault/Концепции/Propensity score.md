---
type: concept
module: М6
tags: [каузальность, наблюдательные]
---
# Propensity score

Propensity score e(X)=P(D=1|X) — вероятность назначения лечения по pre-treatment ковариатам. Истинный e(X) является balancing score: D независимо от X условно по e(X). Оценённый score не гарантирует баланс автоматически.

Для каузального adjustment нужны consistency, (Y(0),Y(1)) ⟂ D | X и overlap. Одного определения P(D|X) недостаточно: условная независимость потенциальных исходов — содержательное допущение об отсутствии оставшегося confounding.

После matching/weighting проверяют распределения ковариат, абсолютные SMD, overlap и ESS. |SMD|<0.1 — ориентир, не доказательство идентификации. SMD после caliper считают по реально оставленным treated и контролям с их весами; исходный SD можно зафиксировать для сопоставимости. Ограничение common support меняет целевую популяцию.

[[IPW]], [[Doubly robust]], [[ATE]]. Практика: [practice_6_4.py](../../course/modules/M6/practice/practice_6_4.py).

Модуль: [[М6 Каузальный вывод]]

## Где почитать
- [[koch-kir — Causal Inference]] (§conditioning)
- [[Kohavi — Trustworthy OCE]] гл.11; [[HKU — Digital Experimentation]] L7
- [[Deng — Causal Inference and Its Applications]] гл.4
- [[stats_for_science]] «Большой матчинг»
