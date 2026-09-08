---
type: concept
module: М6
tags: [каузальность, квазиэксперименты]
---
# DiD

Difference-in-Differences: двойная разность (до/после × тритмент/контроль). Ключевое допущение — параллельные тренды: без лечения группы двигались бы одинаково.

## Зачем
Оценка эффекта роллаутов «на часть групп» (новая фича в одном регионе) без рандомизации; рабочий метод, когда контрольная группа есть, но не случайна.

## Ключевые идеи
- Кейс HKU L7: Seeking Alpha — процедура DiD на реальных данных
- Проверка parallel trends на пре-периоде (визуально + тесты); event study как расширение
- MCP: DiD для роллаутов на части групп с допущением параллельных трендов
- Нобелевка 2021 за натурные эксперименты — DiD/synthetic control/RDD (abba_testing); X5 анонс урока 6 (DiD, synthetic control, event study)
- Практика урока 6.3: DiD на симуляциях (scipy/[[PyMC]]) (план)
- Гостевая лекция HKU: сочетается с PSM (matching + DiD)

Связано: [[Interrupted time series]], [[Synthetic control]], [[RDD]], [[Geo-эксперимент]], [[Propensity score]]

Модуль: [[М6 Каузальный вывод]]

## Где почитать
- [[HKU — Digital Experimentation]] L7
- [[MCP Analytics — Causal Inference Methods]]
- [[abba_testing]] пост про Нобелевку; [[X5 Product Analytics]] анонс урока 6
- [[Kohavi — Trustworthy OCE]] гл.11
