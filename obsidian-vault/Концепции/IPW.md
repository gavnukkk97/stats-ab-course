---
type: concept
module: М6
tags: [каузальность, наблюдательные]
---
# IPW

Inverse probability weighting использует веса D/e(X) для treated и (1−D)/(1−e(X)) для controls при оценке ATE. Для ATT обычно treated имеют вес 1, controls — e(X)/(1−e(X)). Идентификация требует consistency, exchangeability и overlap.

Различайте Horvitz–Thompson и нормированную оценку Hájek. Умножение всех весов одной группы на одну константу, включая числитель стабилизированного веса P(D=d), не меняет её нормированное среднее и ESS=(Σw)²/Σw². Поэтому стабилизация такого вида не исправляет экстремальный относительный разброс.

Транкация и trimming могут улучшить дисперсию ценой смещения или смены estimand. Нужны диагностика overlap, распределения весов и uncertainty с учётом оценки propensity; большой ESS не устраняет скрытый confounding.

[[Propensity score]], [[Doubly robust]], [[ATE]].

Модуль: [[М6 Каузальный вывод]]

## Где почитать
- [[Deng — Causal Inference and Its Applications]] гл.4
- [[koch-kir — Causal Inference]]
- [[shelter_analytics]] «AI в Causal Inference» (AIPW + EconML)
