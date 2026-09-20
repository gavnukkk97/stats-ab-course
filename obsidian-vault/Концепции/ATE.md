---
type: concept
module: М6
tags: [каузальность]
---
# ATE

Average Treatment Effect: E[Y(1)−Y(0)] — средний эффект в явно заданной популяции. ATT усредняет по получившим лечение, CATE — условно по X, SATE — по конечной выборке, SATT — по treated этой выборки. SATT не означает «сегмент после отсечения».

В корректном рандомизированном эксперименте разность средних оценивает эффект назначения в экспериментальной популяции. Перенос на всех пользователей требует дополнительных допущений; при интерференции индивидуальный ATE может не описывать полную раскатку.

Название метода не определяет estimand: matching с caliper может оценивать эффект для оставленных treated; IV — LATE; RDD — эффект у порога; PLR-DML при неоднородности — overlap-взвешенный эффект. AIPW может оценивать ATE при exchangeability, consistency и overlap. Сравнивать оценки разных целевых величин как одну «истину» нельзя.

[[Potential outcomes]], [[Propensity score]], [[Double ML]], [[SUTVA]].

Модуль: [[М6 Каузальный вывод]]

## Где почитать
- [[HKU — Digital Experimentation]] L2–L5
- [[Ci2Lab — Applied Causal Inference]] CH-4
- [[Deng — Causal Inference and Its Applications]] гл.3–4
- [[stats_for_science]] «Большой матчинг»
