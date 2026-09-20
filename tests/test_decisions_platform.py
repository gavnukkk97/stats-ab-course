"""Численные и поведенческие регрессии; учебные скрипты не исполняются при импорте."""
import ast
import copy
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import types
import zlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]

def definitions(relative):
    path=ROOT/relative
    tree=ast.parse(path.read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    namespace=dict(np=np,pd=pd,stats=stats,copy=copy,hashlib=hashlib,math=math,
                   dataclass=dataclass,field=field,zlib=zlib,ALPHA=.05,POWER=.8,
                   N_BUCKETS=1000,LAMBDA_PAIR=.005,Z_A=stats.norm.isf(.025),
                   Z_B=stats.norm.ppf(.8),WEEKLY=11700,DAILY=11700/7,
                   __name__=__name__)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

@pytest.mark.parametrize('p,lo,hi,safe,expected',[
    (.001,-.10,-.02,True,'ROLLBACK'),
    (.001,.04,.08,True,'ROLL —'),
    (.001,.01,.08,True,'HOLD'),
    (.2,-.03,.08,True,'HOLD'),
    (.001,.04,.08,False,'HOLD'),
])
def test_decision_bounds(p,lo,hi,safe,expected):
    fn=definitions('course/modules/M4/practice/practice_4_4.py')['decide']
    assert fn(p,lo,hi,.03,safe).startswith(expected)

def test_ratio_counts_users_without_orders_correctly():
    fn=definitions('course/modules/M4/practice/practice_4_4.py')['ratio_delta']
    late=np.array([0.,1.,1.,0.]);orders=np.array([0.,1.,3.,4.])
    a,b,_,_=fn(late,orders,late,orders)
    assert a==pytest.approx(2/8) and b==a
    assert a!=pytest.approx(np.mean(np.divide(late,orders,out=np.zeros(4),where=orders>0)))

def test_prop_uses_unequal_denominators():
    fn=definitions('course/modules/M4/practice/practice_4_4.py')['prop_z']
    p1,p2,_,_=fn(80,100,180,200)
    assert p1==.8 and p2==.9

def test_sequential_feature_marginals_compose():
    fn=definitions('course/modules/M4/practice/practice_4_5.py')['feature_multiplier']
    effects=np.array([.05,.03,.02,-.01,.07])
    marginals=[fn(effects[:k+1])/fn(effects[:k]) for k in range(len(effects))]
    assert np.prod(marginals)==pytest.approx(fn(effects))

def test_domain_share_and_snapshot_replay():
    ns=definitions('course/modules/M7/practice/practice_7_1.py');service=ns['SplitService'](250)
    service.add_domain('half',.5);service.add_layer('rank','half','rank');service.add_experiment('rank','test',1)
    ids=[f'u{i}' for i in range(5000)]
    before=[service.assign('rank',x) for x in ids]
    assert .45<np.mean([x is not None for x in before])<.55
    replay=ns['SplitService'].from_snapshot(service.snapshot())
    assert [replay.assign('rank',x) for x in ids]==before
    service.domains['half']['traffic']=1
    assert replay.domains['half']['traffic']==.5

def test_assignment_has_both_variants_with_custom_bucket_count():
    ns=definitions('course/modules/M7/practice/practice_7_1.py');service=ns['SplitService'](16)
    service.add_domain('all');service.add_layer('rank','all');service.add_experiment('rank','test',1)
    assert {service.assign('rank',str(i)) for i in range(200)}=={'test:A','test:B'}

def test_bucket_sufficient_statistics_match_user_welch():
    fn=definitions('course/modules/M7/practice/practice_7_2.py')['welch_from_summaries']
    a=np.array([0.,1.,3.,20.,8.]);b=np.array([2.,3.,5.,7.,9.,11.])
    def summary(x):
        parts=[x[:1],x[1:]]
        return [len(y) for y in parts],[y.sum() for y in parts],[(y*y).sum() for y in parts]
    lift,t,p=fn(*summary(a),*summary(b))
    reference=stats.ttest_ind(b,a,equal_var=False)
    assert t==pytest.approx(reference.statistic)
    assert p==pytest.approx(reference.pvalue)
    assert lift==pytest.approx(b.mean()/a.mean()-1)

@pytest.mark.parametrize('reason',['SRM: missing','ИНВАРИАНТ: changed','GUARDRAIL: harm'])
def test_every_stop_blocks_shipping_even_on_final_day(reason):
    platform=definitions('course/modules/M7/practice/practice_7_3.py')['ExperimentPlatform']()
    run=types.SimpleNamespace(stop_reason=reason,stopped_early=False)
    result=platform.verdict({},run,{'oec':{'est':.1,'p':.001},'guardrail':{'safe':True,'breach':False}})
    assert not result.startswith('SHIP')

def test_slots_reserved_and_instances_isolated():
    ns=definitions('course/modules/M7/practice/practice_7_3.py');a,b=ns['ExperimentPlatform'](),ns['ExperimentPlatform']()
    doc=ns['DesignDoc'](exp_id='one',name='one',layer='ранжирование',oec='конверсия в заказ (2 нед.)',mde_rel=.1,share=.6)
    assert a.submit(doc) is not None
    assert b.LAYERS['ранжирование']==[]
    doc2=ns['DesignDoc'](exp_id='two',name='two',layer='ранжирование',oec=doc.oec,mde_rel=.1,share=.5)
    assert a.submit(doc2) is None
    a.release('one');assert a.submit(doc2) is not None

def test_capstone_preserves_assignment_totals():
    base=ROOT/'course/capstone/data'
    assignments=pd.read_csv(base/'assignments.csv');events=pd.read_csv(base/'events.csv')
    assert assignments.user_id.is_unique
    assert events.event_id.duplicated().any()
    clean=events.drop_duplicates('event_id')
    units=assignments.merge(clean.groupby('user_id').margin_kopecks.sum(),on='user_id',how='left').fillna({'margin_kopecks':0})
    assert len(units)==len(assignments)==6000
    assert units.margin_kopecks.sum()==clean.margin_kopecks.sum()
    assert (units.margin_kopecks==0).any()


def test_total_sample_mde_has_two_arms():
    fn=definitions('course/modules/M7/practice/practice_7_1.py')['mde_for_total']
    assert fn(2_000_000,2.5)==pytest.approx(.0099050953649,rel=1e-5)
    assert fn(500_000,2.5)==pytest.approx(2*fn(2_000_000,2.5))
