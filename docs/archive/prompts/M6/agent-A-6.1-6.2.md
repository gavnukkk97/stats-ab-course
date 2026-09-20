Ты пишешь контент уроков курса «Проверка гипотез: статистика, АБ-тесты, каузальный вывод». Модуль М6 «Каузальный вывод», уроки 6.1 и 6.2 — вступительные. Сквозной кейс: «ЕдаДома». Студенты прошли М1–М5 (вся статистика, АБ, байес).

ВАЖНО: в course/content/M6/ уже лежат practice_6_1.py и practice_6_2.py от предыдущего запуска (проверь их: если запускаются exit 0 и покрывают темы — используй как есть, только подставь их реальные числа в уроки; если битые/не те — перепиши). Уроки lesson-6.1.md и lesson-6.2.md НЕ существуют — пиши с нуля. НЕ трогай M6/visuals/.

СНАЧАЛА ПРОЧИТАЙ:
- /Users/user/.zcode/workspace/default/course/plan.md — строки уроков 6.1, 6.2
- /Users/user/.zcode/workspace/default/materials/local/causal-inference-doc.md — секции 1–4 (SCM vs PO, конфаундер/коллайдер/медиатор, backdoor/frontdoor)
- /Users/user/.zcode/workspace/default/materials/telegram/telegraph-causal-series.md — «CI на пальцах» ч.1–2 (интуиции)
- /Users/user/.zcode/workspace/default/materials/local/applied-causal-inference-course.md — Ci2Lab CH-1–3
- /Users/user/.zcode/workspace/default/materials/web/alexdeng-causal-book.md — гл.1–2, 4–5
- /Users/user/.zcode/workspace/default/materials/local/lecture-catch-effect.md — лекция пользователя (сл.14–15, 19–20)

ФОРМАТ (course/content/M6/lesson-6.1.md, 6.2):
frontmatter (module: М6, lesson, title) → # Урок → **Цель** → ## Зачем аналитику → ## Теория (4-6 подразделов, формулы, интуиция, считаемые примеры) → ## Разбор на данных → ## Подводные камни (5-7) → ## Практика → ## ДЗ (4-5, <details>) → ## Шпаргалка → ## Источники. Маркеры `<!-- VIS: описание -->` отдельной строкой (картинки v72–v83 уже лежат в visuals/ — не вставляй сам).

ТЕМЫ:
- 6.1 «Корреляция ≠ причинность: конфаундеры, коллайдеры, DAG»: три класса объяснений корреляции; fork/chain/collider на продуктовых примерах; d-разделение интуицией; Симпсон (числовой пример) и Берксон; «корректировка за коллайдер» — классическая ошибка; DAG как рабочий инструмент (рисуем до анализа); «causal salad».
- 6.2 «Potential outcomes»: Y(1)/Y(0), фундаментальная проблема; ATE/ATT/CATE/ITE — таблица «вопрос→эстиманд» (+compliers/LATE вскользь); почему «до/после» ≠ эффект; SUTVA (no interference + consistency) и где ломается в продукте; рандомизация уравнивает группы «в среднем по всему, включая ненаблюдаемое» — мостик к тому, почему АБ золотой стандарт; ignorability — что нужно без рандомизации (форвард 6.4).
Практики (если нужно переписать): practice_6_1.py — Симпсон на данных + collider-симуляция (ложная корреляция при conditioning) + мини-DAG-утилита текстом; practice_6_2.py — симуляция PO: наивная разность vs истинный ATE при конфаундере; рандомизация чинит; SUTVA-нарушение (spillover) смещает; ATT≠ATE при self-selection.
ОБЯЗАТЕЛЬНО запусти практики, exit 0.

ВЕРНИ КОРОТКО: пути, состав, результат запуска, VIS-маркеры.