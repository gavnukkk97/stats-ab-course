---
type: source
kind: paper
module: [М3, М4]
tags: [аб, cupac, ml, чувствительность, paper]
---
# DoorDash — CUPAC

Jeff Li (DoorDash) «Improving Experimental Power through Control Using Predictions as Covariate (CUPAC)» (2020) — блог-пост, закрепивший термин CUPAC.

Чем ценен: определение и правила ML-ковариат: контрольная величина — предсказание метрики Y моделью (GBM на сотнях фичей), обученной на данных, не затронутых экспериментом; на периоде теста только инференс; цель — максимум R² по Y; cross-fitting. Результаты: снижение неопределённости оценки эффекта, детект меньших эффектов; в switchback-экспериментах логистики (Tang et al. 2020) — −25%+ длительности. Опасность: переобучение/утечки фичей → bias, нужен A/A-контроль.

Закрывает: [[CUPAC]], [[ML variance reduction]], [[CUPED]], [[Switchback-дизайн]].

Оригинал: https://careersatdoordash.com/blog/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/ ; конспект — `Материалы/research-sensitivity.md` §6.

Модули: [[М3 Статистика для АБ]] (урок 3.5), [[М4 АБ на практике]].
