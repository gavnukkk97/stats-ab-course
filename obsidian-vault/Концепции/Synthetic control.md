---
type: concept
module: М6
tags: [каузальность, квазиэксперименты]
---
# Synthetic control

Synthetic control строит контрфакт treated-юнита как взвешенную комбинацию незатронутых доноров. В классическом варианте веса неотрицательны и суммируются в 1; их подбирают по pre-периоду и предикторам.

Необходимы подходящие доноры, достаточно pre-данных, отсутствие spillover/anticipation и переносимость восстановленной связи на post. Хороший pre-fit сам по себе не исключает отдельного post-шока treated. Без доноров synthetic control неприменим; ITS требует других допущений.

Placebo по юнитам и датам диагностирует дизайн. Ранг эффекта среди placebo является randomization p-value только при соответствующей exchangeability механизма назначения и симметричной процедуре; в обычной observational задаче это сравнительная диагностика.

Практика [practice_6_3.py](../../course/modules/M6/practice/practice_6_3.py): constrained веса и placebo-rank с оговоркой. [[DiD]], [[Interrupted time series]].

Модуль: [[М6 Каузальный вывод]]

## Где почитать
- [[shelter_analytics]] Кейс 0 (контрфактическая модель выручки)
- [[MCP Analytics — Causal Inference Methods]]
- [[koch-kir — Causal Inference]]; [[abba_testing]] кейс маржи
- План [[М6 Каузальный вывод]] урок 6.3
