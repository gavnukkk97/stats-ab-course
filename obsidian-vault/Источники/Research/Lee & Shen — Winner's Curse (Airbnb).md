---
type: source
kind: paper
module: [М4]
tags: [аб, winner's-curse, paper]
---
# Lee & Shen — Winner's Curse (Airbnb)

Lee & Shen (Airbnb) «Winner's Curse: Bias Estimation for Total Effects of Features in Controlled Experiments» (KDD 2018).

Чем ценен: каноническая формулировка проклятия победителя для A/B-платформ: фичи запускаются по критерию значимости → отобраны те, у кого наблюдаемый эффект наибольший → сумма оценок смещена вверх. Даёт несмещённую оценку суммарного эффекта + CI. Продолжение линии — Deng «On Post-selection Inference in A/B Testing» (2019); операция — Gelman & Carlin Type S / Type M errors. Парные источники: байесовский shrinkage (arXiv 2608.12949), GrowthBook «Why Summing Your Experiment Wins Overstates Impact».

Закрывает: [[Winner's curse]], [[Кумулятивные эффекты фич]], [[Холдаут-группа]].

Оригинал: https://dl.acm.org/doi/10.1145/3219819.3219905 ; конспект — `Материалы/research-gaps.md` §Задача 3.

Модуль: [[М4 АБ на практике]] (урок 4.5).
