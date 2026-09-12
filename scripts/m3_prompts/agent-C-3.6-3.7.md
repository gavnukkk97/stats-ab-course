Ты пишешь контент уроков курса «Проверка гипотез: статистика, АБ-тесты, каузальный вывод». Модуль М3, уроки 3.6 и 3.7. Сквозной кейс: «ЕдаДома». Студенты знают М1–М2 и (параллельно пишутся, считай что знают): 3.1 MDE/мощность, 3.3 ratio/линеаризация, 3.4 винзоризация, 3.5 CUPED/стратификация.

СНАЧАЛА ПРОЧИТАЙ:
- /Users/user/.zcode/workspace/default/course/plan.md — строки уроков 3.6, 3.7
- /Users/user/.zcode/workspace/default/materials/web/research-sensitivity.md — ГЛАВНЫЙ источник для 3.6: CUPED-оригинал (Deng/Xu/Kohavi/Walker WSDM'13), CUPAC DoorDash, Yandex BDT, Meta, Lin 2013, Jeunen unified view
- /Users/user/.zcode/workspace/default/materials/web/alexdeng-causal-book.md — гл.10 (sensitivity = power × распределение эффектов, variance reduction)
- /Users/user/.zcode/workspace/default/materials/telegram/abba_testing-digest.md — серия mSPRT ×5 частей (главный источник 3.7), «Точки над CUPED»
- /Users/user/.zcode/workspace/default/materials/telegram/stats_for_science-digest.md — серия sequential testing ×3
- /Users/user/.zcode/workspace/default/materials/telegram/trisigma_avito-digest.md — кейс Авито CUPED/CUPAC «в пару дюжин раз быстрее»
- /Users/user/.zcode/workspace/default/materials/web/research-arxiv.md — разделы про anytime-valid/sequential

ФОРМАТ (course/content/M3/lesson-3.6.md, 3.7; папка M3 существует; НЕ трогай M3/visuals/):
frontmatter → # Урок → **Цель** → ## Зачем аналитику → ## Теория → ## Разбор на данных → ## Подводные камни (5-7) → ## Практика → ## ДЗ (4-5, <details>) → ## Шпаргалка → ## Источники. Маркеры `<!-- VIS: описание -->`.

ТЕМЫ:
- **3.6 «Продвинутое снижение дисперсии: ML-ковариаты и регрессионная поправка»**: от CUPED (одна ковариата) к ML: CUPAC (DoorDash) — предсказание метрики по пре-данным как ковариата; Yandex BDT-поправка; почему ML-предсказание легально (строится ТОЛЬКО на пре-периоде); Lin 2013: центрирование ковариат + взаимодействия T×X + robust SE → «нельзя ухудшить difference-in-means» (связь с прод-CUPED Microsoft ExP); Jeunen: CUPED/CUPAC/ML-RATE как частные случаи doubly robust регрессии; чувствительность = мощность × распределение истинных эффектов (Deng гл.10); когда CUPED НЕ помогает (нет пре-данных, новые юзеры, слабая ρ); кейс Авито (trisigma): связка методов «в пару дюжин раз быстрее». Практика (practice_3_6.py): синтетика с несколькими пре-ковариатами: CUPED (1 ковариата) vs CUPAC (numpy lstsq по нескольким ковариатам как «ML») vs Lin-оценка; сравнить SE/CI; показать ошибку: ковариата, построенная на данных эксперимента → смещение (симуляция); график «экономия дней теста» от ρ².
- **3.7 «Подглядывание и последовательное тестирование»**: peeking: почему «смотреть каждый день и остановиться при p<0.05» раздувает α до 20-30%+ (симуляция обязательна); почему продление теста после «прокраса» не чинит (Atlamos); решения: (а) group sequential (альфа-спендинг, O'Brien-Fleming/Pocock — идея), (б) mSPRT/always-valid p-value (серия abba ×5: механика доступно — mixture over λ, смешанное SPRT, растущая граница), (в) байесовский взгляд (форвард М5 — «можно останавливаться когда угодно» с оговорками); цена гибкости: чуть шире интервалы / больше n; когда sequential оправдан (длинные тесты, дорогие фичи, раннее обнаружение вреда = guardrails). Практика (practice_3_7.py): симуляция: ежедневный peeking в A/A — эмпирическая α vs номинал (кривая инфляции от числа проверок); упрощённая mSPRT-подобная always-valid статистика на нормальных данных — показать, что A/A-уровень держится при ежедневных проверках; сравнить: сколько дней до детекции фиксированного эффекта fixed vs sequential.

ПРАКТИКИ: percent-формат, numpy/scipy/matplotlib/pandas (БЕЗ sklearn), seed, Agg, png в M3/, print. ОБЯЗАТЕЛЬНО запусти оба, exit 0.

ВЕРНИ КОРОТКО: пути к 4 файлам, состав, результат запуска, VIS-маркеры.
