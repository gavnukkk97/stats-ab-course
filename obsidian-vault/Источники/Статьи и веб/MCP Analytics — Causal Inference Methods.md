---
type: source
kind: article
module: [М6, М5]
tags: [каузальность, байес, статья]
---
# MCP Analytics — Causal Inference Methods

MCP Analytics, «Causal Inference Methods: A/B Testing, DiD, Propensity Scores & More» — обзор 9 методов каузального вывода с условиями применимости.

Чем ценен: карта методов для урока 6.5: A/B (золотой стандарт при контроле и рандомизации), байесовский A/B (вероятностные утверждения «94% что B лучше A минимум на 3%», ранняя остановка, цена — априор), [[DiD]] (параллельные тренды), [[Propensity score|PSM]] (нужны богатые ковариаты), [[Инструментальная переменная|IV]] (плохой инструмент «хуже OLS»), [[RDD]] (нужны плотные данные у порога), Causal Impact (Google, байесовские временные ряды), [[Synthetic control]] (прозрачен для стейкхолдеров), ANCOVA (=[[CUPED]]). Иерархия валидности: рандомизация → квазиэксперименты → матчинг.

Закрывает: [[DiD]], [[RDD]], [[Synthetic control]], [[Interrupted time series]], [[Propensity score]], [[Инструментальная переменная]], [[Байесовский AB-тест]], [[CUPED]], [[Каузальность]].

Оригинал: https://mcpanalytics.ai/articles/causal-inference-methods ; конспект в `Материалы/web-articles.md` §5.2.

Модули: [[М6 Каузальный вывод]] (уроки 6.3, 6.5), [[М5 Байесовские методы]] (§байес A/B).
