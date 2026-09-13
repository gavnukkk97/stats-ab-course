Ты пишешь контент ключевого урока курса «Проверка гипотез: статистика, АБ-тесты, каузальный вывод». Модуль М6, урок 6.3 «Квазиэксперименты» — центральный. Сквозной кейс: «ЕдаДома». Студенты знают М1–М5 и параллельные 6.1–6.2 (DAG, PO, ATE/ATT).

ВАЖНО: в course/content/M6/ лежит practice_6_3.py от прошлого запуска — проверь (exit 0, покрывает темы — используй; битые — перепиши). Урок lesson-6.3.md не существует — пиши с нуля. НЕ трогай M6/visuals/.

СНАЧАЛА ПРОЧИТАЙ:
- /Users/user/.zcode/workspace/default/course/plan.md — строка 6.3
- /Users/user/.zcode/workspace/default/materials/web/research-did-psm.md — ГЛАВНЫЙ: DR-DiD Sant'Anna&Zhao, генеалогия, LaLonde-лестница, pretrend-тесты, грабли, DGP для практики
- /Users/user/.zcode/workspace/default/materials/local/causal-inference-doc.md — секции 9, 11–12 (DiD + staggered TWFE + CS, RDD, ITS, SC+TRoP)
- /Users/user/.zcode/workspace/default/materials/local/lecture-catch-effect.md — лекция пользователя (сл.15–22: карта методов, кейс тренинга DiD+matching)
- /Users/user/.zcode/workspace/default/materials/telegram/telegraph-causal-series.md — ч.4: DiD Пекин/Wal-Mart, RDD стипендия/алкоголь-21, IV encouragement Spotify, DiD Practitioner's Guide
- /Users/user/.zcode/workspace/default/materials/web/web-articles.md — MCP Analytics, koch-kir causal

ФОРМАТ (course/content/M6/lesson-6.3.md): frontmatter → # Урок → **Цель** → ## Зачем аналитику → ## Теория (6 подразделов, см. ниже) → ## Разбор на данных → ## Подводные камни (6-8) → ## Практика → ## ДЗ (4-5, <details>) → ## Шпаргалка → ## Источники. Маркеры `<!-- VIS: описание -->`.

СТРУКТУРА ТЕОРИИ:
1. Pre-Post: почему слаб (30 секунд)
2. DiD: 2×2, параллельные тренды; регрессия Post×Treat; event study для pretrends; staggered TWFE-ловушка (плохие контрали) + Callaway&Sant'Anna; плацебо-периоды
3. DiD+PSM: матчинг выравнивает группы → conditional parallel trends; кейс тренинга; DR-DiD (SZ'20: состоятелен при верной одной из моделей); ЧЕСТНО: когда комбинация не лучше (Chabé-Ferret: regression to the mean, Ashenfelter's dip); цена +35–67% SE, common support
4. RDD: running variable + порог; sharp/fuzzy; проверки (плотность у порога/McCrary, плацебо-пороги, ковариаты); кейсы стипендии/алкоголь-21
5. ITS + Causal Impact: тренд+сезонность+сдвиг, BSTS «прогноз как контрфакт»
6. Synthetic control: смесь контролей = псевдо-тритмент; один юнит; pretrend-веса; TRoP/placebo-юниты
ПРАКТИКА (practice_6_3.py — проверь/допиши): DiD 2×2 руками + numpy-lstsb-регрессия; pretrend event study (нули под H0); staggered-ловушка: naive TWFE врёт, групповая DiD чинит; synthetic control (веса lstsq по претренду); мини-кейс DiD+PSM по DGP из research-did-psm (панель ~1500, 6 периодов, конфаундер, истинный τ; лестница naive→+X→PSM(логистика+ближайший сосед)→DR; грабля с post-treatment ковариатой). БЕЗ statsmodels (OLS через numpy).
ОБЯЗАТЕЛЬНО запусти, exit 0.

ВЕРНИ КОРОТКО: путь, состав, результат, VIS-маркеры.