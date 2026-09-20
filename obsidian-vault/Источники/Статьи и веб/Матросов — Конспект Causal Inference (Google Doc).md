---
type: source
kind: article
module: М6
tags: [causal-inference, конспект]
---
# Матросов — Конспект Causal Inference (Google Doc)

Живой русский конспект по каузальному выводу от Сергея Матросова (@smatrosov, канал [[abba_testing]], X5). ~120+ стр., 2336 абзацев, 63 таблицы с ручными расчётами; ведётся с 03.2026. Главный тезис: «если можно A/B — делай A/B; CI — вынужденная мера для observed data».

## Чем ценен (самый глубокий источник по М6)
- **Очень глубоко:** PSM/matching + валидация (SMD, Love plot, L1, ESS, A/A на предпериоде), IV (Wald с выводом, 2SLS, F>10, monotonicity), DiD со staggered TWFE-ловушкой (Callaway & Sant'Anna), sensitivity-анализ (гамма Розенбаума, дельта Остера, E-value, Sensemakr), refutation/плацебо-тесты, CATE-модели (Causal Forest, S/T/X/R-Learners).
- **Средне:** DAG (backdoor/frontdoor без формального do-calculus), causal discovery (GES, PC, LiNGAM), Double ML, synthetic control.
- **Базово:** RDD, ITS, Causal Impact.
- Форма: текст + числовые мини-датасеты, ключевые оценки посчитаны руками — готовые демо для уроков.

## Закрывает концепции
[[Propensity score]], [[IPW]], [[Инструментальная переменная]], [[DiD]], [[Synthetic control]], [[Doubly robust]], [[Double ML]], [[Potential outcomes]], [[Конфаундер]], [[Коллайдер]], [[DAG]], [[ATE]], [[RDD]], [[Interrupted time series]]

## Оригинал
- Полный текст: авторский конспект не включён в репозиторий; доступные первоисточники перечислены в [указателе курса](../../../course/SOURCES.md)
- Исходник: Google Doc «Causal Inference» из канала [[abba_testing]] (файл Causal Inference.docx)
