"""Рабочий файл ученика. Не открывайте solutions до собственной интерпретации."""
from pathlib import Path
import pandas as pd

DATA=Path(__file__).resolve().parent/'data'

def load_data():
    return {name: pd.read_csv(DATA/f'{name}.csv') for name in ('assignments','exposures','events')}

def audit_and_build_units(data):
    """TODO: проверки ключей, назначения, окон; пользовательская таблица с нулями."""
    raise NotImplementedError('Сначала запишите популяцию и правила качества в протоколе')

def analyze(units):
    """TODO: OEC, CI, p, допустимый вред guardrail; решение отдельно от расчёта."""
    raise NotImplementedError('Определите estimands и метод неопределённости')

if __name__=='__main__':
    data=load_data()
    for name, frame in data.items():
        print(name,frame.shape,list(frame.columns))
    # Раскомментируйте после заполнения функций:
    # print(analyze(audit_and_build_units(data)))
