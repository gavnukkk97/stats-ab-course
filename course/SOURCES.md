# Проверяемые источники

Локальные `materials/*` в старых библиографиях — рабочие конспекты авторов, не публичные учебные зависимости. Для воспроизводимого утверждения используйте первоисточник, название раздела/таблицы и дату версии. Ниже точки входа; дополнительные прямые ссылки сохранены в уроках и кейсах.

| Тема | Источник | Как использовать |
|---|---|---|
| Статистика экспериментов | [Alex Deng, Causal Inference and Its Applications](https://alexdeng.github.io/causal/) | формулы и допущения; отличать теорему от эвристики |
| p-value | [ASA Statement](https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf) | смысл p, ограничения порогового решения |
| Mann–Whitney | [SciPy mannwhitneyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html) | H0, ties и exact/asymptotic методы |
| Bootstrap | [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) | виды интервалов и интерфейс |
| FDR | [SciPy false_discovery_control](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html) | BH/BY и зависимость |
| Регрессионная корректировка | [Lin, 2013](https://arxiv.org/abs/1208.2301) | асимптотические гарантии и условия |
| ML-adjustment | [MLRATE](https://arxiv.org/abs/2106.07263) | cross-fitting и оценивание эффекта |
| Адаптивные данные | [Hadad et al.](https://arxiv.org/abs/1911.02768) | специальные методы inference после адаптивного назначения |
| Matching и bootstrap | [Abadie & Imbens](https://www.nber.org/papers/t0325) | почему обычный bootstrap не универсален |
| Чувствительность | [Rosenbaum, sensitivitymv](https://cran.r-project.org/web/packages/sensitivitymv/sensitivitymv.pdf) | определения границ, а не шкала доверия |
| RDD | [Cattaneo & Titiunik](https://rdpackages.github.io/references/Cattaneo-Titiunik_2022_ARE.pdf) | continuity/local randomization и локальный эффект |
| E-value | [Ding & VanderWeele](https://arxiv.org/abs/1507.03984) | совместная граница силы двух связей |
| Triggering | [Microsoft, pre-experiment](https://www.microsoft.com/en-us/research/?p=680556) | counterfactual logging и eligible-популяция |
| eBay | [Blake, Nosko, Tadelis](https://faculty.haas.berkeley.edu/stadelis/Tadelis.pdf) | различать brand/non-brand эксперименты |
| LLM-претесты | [AgentA/B v4](https://arxiv.org/html/2504.09723v4) | предварительное исследование; числовая несогласованность обсуждается в 4.6 |
| Интеграции платформ | [Optimizely/Jira](https://docs.developers.optimizely.com/web-experimentation/docs/configure-jira) | сверять актуальность продукта/тарифа на дату выбора |

Пересказ Telegram-дискуссии без данных не позволяет самостоятельно установить false positive. Синтетические примеры показывают свойства заданной модели; их численные эффекты не являются измерениями реальной компании. При недоступном источнике обозначайте неподтверждённое утверждение и не достраивайте его точными числами.
