"""Generate a design map without unsupported universal method rankings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "skills" / "method-roadmap" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
fig, ax = plt.subplots(figsize=(12,6)); ax.axis('off')
rows = [
 ['Рандомизация доступна', 'A/B / cluster / switchback', 'Назначение, протокол, интерференция'],
 ['Контроль + данные до', 'DiD; при необходимости DR-DiD', 'Контрфактические параллельные тренды'],
 ['Правило назначения по порогу', 'RDD', 'Непрерывность; локальный estimand'],
 ['Незатронутые доноры + pre', 'Synthetic control', 'Pre-fit и переносимость связи'],
 ['Достаточные pre-X + overlap', 'Matching / IPW / DR', 'Conditional exchangeability'],
 ['Допустимый внешний инструмент', 'IV', 'Relevance, independence, exclusion; LATE'],
 ['Обоснованная экстраполяция ряда', 'ITS / прогнозный контроль', 'Нет одновременного шока; стабильность'],
]
table=ax.table(cellText=rows,colLabels=['Что доступно', 'Кандидат', 'Что обосновать'],cellLoc='left',loc='center',colWidths=[.34,.30,.36])
table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2.7)
ax.set_title('Вопрос → estimand → идентификация → оценка и неопределённость',fontsize=15,pad=20)
fig.text(.5,.025,'Несколько дизайнов могут подходить одновременно. Универсального ранга доверия нет.',ha='center',fontsize=11)
fig.tight_layout(rect=(0,.04,1,1));dest=OUTPUT_DIR/'decision_roadmap.png';fig.savefig(dest,bbox_inches='tight');plt.close(fig)
causal_images = ROOT / 'skills' / 'causal-estimation' / 'images'
causal_images.mkdir(parents=True, exist_ok=True)
shutil.copyfile(dest, causal_images / 'decision_roadmap.png')
print('Готово:', dest, 'и', causal_images / 'decision_roadmap.png')
