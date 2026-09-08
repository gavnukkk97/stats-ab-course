---
type: concept
module: М3
tags: [аб, чувствительность, ml]
---
# ML variance reduction

Семейство методов, где снижение дисперсии в A/B достигается ML-моделью: предсказание метрики (CUPAC, Yandex BDT, LLM-«двойники»), предсказание дисперсии юнита (Meta variance-weighted), нелинейная поправка (Lin + ML). Общая рамка: Var(δ_VR) = Var(δ_DiM)·(1 − R²).

## Зачем
«Мы управляем σ» — главный рычаг чувствительности (research-sensitivity, план урока 3.5): чувствительность метрики = power × распределение истинных эффектов; чем точнее прогноз, тем меньше «неопределённость оценки эффекта» и короче тест.

## Ключевые идеи
- Лестница методов: CUPED (линейная поправка) → BDT/CUPAC (бустинг/GBDT) → гибрид pre+in-experiment (Etsy 2410.09027; Deng KDD'23 τ-adjustment) → variance-weighted (Meta: −17%, с CUPED ≈ −50%) → LLM-цифровые двойники с гарантией «do no harm» (arXiv 2606.08853, KDD'26)
- Правильный CUPED = Lin-оценщик: взаимодействия T×X, центрирование, робастные SE → «adjustment ничего не стоит» (Lin 2013; ExP Deep Dive: раздельные коэффициенты по рукам = ANCOVA2)
- Унификация: DiM ≡ IPS с оптимальным control variate; CUPED/CUPAC/ML-RATE ≡ doubly robust (Jeunen 2603.08370) — мост к [[Double ML]]
- Платформенная метрика качества VR — «effective traffic multiplier» 1/(1−R²) (Microsoft ExP); выигрыш неравномерен: считать на метрику, не «в среднем»
- Практика Etsy: −66% медианной дисперсии; Instacart: power analysis с VR-поправкой до запуска

Связано: [[CUPED]], [[CUPAC]], [[Doubly robust]], [[Double ML]], [[Мощность]]

Модуль: [[М3 Статистика для АБ]]

## Где почитать
- [[research-sensitivity]] — весь файл (29 источников, детальные конспекты §2, 3, 14 + раздел про Lin/Freedman)
- [[research-arxiv]] — «современный слой» 2023–2026 (строки 1–9 таблицы)
- [[Lin — Agnostic Notes (2013)]], [[Poyarkov et al — BDT adjustment (Yandex)]]
- [[Deng — Causal Inference and Its Applications]] гл.10
