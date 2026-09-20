"""Синтетический учебный датасет; просмотр раскрывает условия задачи."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]/'data'
OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(20260920)
n=6000
activity=rng.gamma(2, .5, n)
variant=rng.choice(['A','B'],n)
pre=rng.poisson(2*activity)*rng.integers(800,1601,n)
assign=pd.DataFrame({'user_id':np.arange(n),'variant':variant,'assigned_day':0,'pre_margin_kopecks':pre})
assign.to_csv(OUT/'assignments.csv',index=False)

# Контакт зависит от pre-activity; treatment его не меняет. ITT остаётся вся assignment-популяция.
contact=rng.random(n)<(.55+.25*np.minimum(activity,2)/2)
ex=pd.DataFrame({'user_id':np.flatnonzero(contact),'variant':variant[contact],'day':rng.integers(0,3,contact.sum())})
ex.insert(0,'exposure_id',np.arange(len(ex)))
retries=ex.loc[rng.random(len(ex))<np.where(ex.variant=='B',.10,.02)]
pd.concat([ex,retries],ignore_index=True).to_csv(OUT/'exposures.csv',index=False)

rows=[];event_id=0
for i in range(n):
    orders=rng.poisson(2*activity[i]*(1+.025*(variant[i]=='B')*contact[i]))
    for _ in range(orders):
        cancelled=rng.random()<.035
        rows.append((event_id,i,int(rng.integers(0,14)),0 if cancelled else int(rng.integers(800,1601)),int(cancelled)))
        event_id+=1
events=pd.DataFrame(rows,columns=['event_id','user_id','day','margin_kopecks','cancelled'])
pd.concat([events,events.sample(frac=.012,random_state=1)],ignore_index=True).to_csv(OUT/'events.csv',index=False)
print(f'Сохранено {n} назначений, {len(ex)} уникальных экспозиций, {len(events)} заказов.')
