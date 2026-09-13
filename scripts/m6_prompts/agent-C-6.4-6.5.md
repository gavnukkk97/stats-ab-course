Ты пишешь контент уроков курса «Проверка гипотез: статистика, АБ-тесты, каузальный вывод». Модуль М6, уроки 6.4 и 6.5 (завершающие). Сквозной кейс: «ЕдаДома». Студенты знают М1–М5 и параллельные 6.1–6.3 (DAG, PO, квазиэксперименты с DiD+PSM).

ВАЖНО: в course/content/M6/ лежит practice_6_4.py от прошлого запуска (проверь: exit 0 и покрывает темы — используй; иначе перепиши). practice_6_5.py НЕ существует — напиши. Уроки lesson-6.4.md, lesson-6.5.md не существуют — пиши с нуля. НЕ трогай M6/visuals/.

СНАЧАЛА ПРОЧИТАЙ:
- /Users/user/.zcode/workspace/default/course/plan.md — строки 6.4, 6.5
- /Users/user/.zcode/workspace/default/materials/local/causal-inference-doc.md — секции 5 (IV: Wald с выводом, 2SLS, F>10, monotonicity), 7 (PSM: caliper, критика Кинга→CEM/MDM), 8 (DML/DR, Causal Forest, S/T/X/R-Learners), 10 (refutation-плацебо ×4, гамма Розенбаума, дельта Остера, E-value)
- /Users/user/.zcode/workspace/default/materials/web/web-articles.md — koch-kir «Causal Inference from Observational Data»
- /Users/user/.zcode/workspace/default/materials/web/ppilif-lectures.md — цикл «Эээээксперименты» (ATE/SUTVA/uplift/matching)
- /Users/user/.zcode/workspace/default/materials/local/lecture-catch-effect.md — лекция пользователя (сл.19–24: matching-двойники, IV расстояние-до-офиса, таблица выбора метода, «комбинация надёжнее»)

ФОРМАТ (course/content/M6/lesson-6.4.md, 6.5): frontmatter → # Урок → **Цель** → ## Зачем аналитику → ## Теория → ## Разбор на данных → ## Подводные камни (5-7) → ## Практика → ## ДЗ (4-5, <details>) → ## Шпаргалка → ## Источники. Маркеры `<!-- VIS: описание -->`.

ТЕМЫ:
- 6.4 «Наблюдательные данные: матчинг, IPW, IV, double ML»: matching (NN, caliper); propensity (логистическая); ЧЕСТЬ: SMD<0.1, Love plot, common support/ESS, критика Кинга (PEARLS→CEM/MDM); IPW + экстремальные веса (стабилизация); doubly robust; IV: эндогенность, три условия+monotonicity, Wald руками (по CI-doc), 2SLS идея, слабый инструмент F<10, LATE; encouragement design (Spotify); double ML (Фриш–Вауг–Ловелл интуиция); CATE-зоопарк (S/T/X/R, Causal Forest) — связь с сегментами 4.4; библиотеки DoWhy/EconML/CausalML (когда какие).
- 6.5 «Доверие к результату и карта методов» (финал модуля): культура refutation (плацебо-исходы/времена/подгруппы, отрицательные контроли); sensitivity: гамма Розенбаума (насколько сильный конфаундер убьёт вывод), E-value (формула+интерпретация), дельта Остера; иерархия внутренней валидности (рандомизация→квази→корректировки); ошибка III рода: прокси-метрики, Гудхарт, медиация; финальное ДЕРЕВО ВЫБОРА МЕТОДА (сл.21 лекции + весь модуль: можно рандомизировать? → этика/стоимость/уже произошло → есть контрольная группа? → параллельные тренды? → порог? → один юнит? → ковариаты); «когда НЕ делать выводов».
ПРАКТИКИ: practice_6_4.py (проверь/допиши): PSM-пайплайн на синтетике с известным эффектом: propensity (scipy/numpy), matching+caliper, Love plot (SMD до/после), лестница наивный→регрессия→PSM→IPW→DR к истине; грабля с post-treatment ковариатой; IV-синтетика: encouragement→Wald≈LATE, слабый инструмент→Wald взрывается.
practice_6_5.py (новый): плацебо-тесты для эффекта из 6.4 (плацебо-исход, плацебо-время); E-value + упрощённая гамма Розенбаума для своего результата; функция method_tree(ситуация) — текстовое дерево; чек-лист «каузальный аудит».
ОБЯЗАТЕЛЬНО запусти оба, exit 0.

ВЕРНИ КОРОТКО: пути, состав, результат, VIS-маркеры.