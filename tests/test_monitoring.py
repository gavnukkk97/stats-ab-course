"""Checks for missingness and the complete monitoring/terminal-analysis workflow."""
import ast
from pathlib import Path
from unittest.mock import patch
import types
import numpy as np
import pytest

P=Path(__file__).resolve().parents[1]/'course/modules/M7/practice/practice_7_3.py'

def load_definitions():
    tree=ast.parse(P.read_text())
    selected=[]
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef)):
            selected.append(node)
        elif isinstance(node,ast.Assign):
            names=[child.id for target in node.targets for child in ast.walk(target) if isinstance(child,ast.Name)]
            if names and all(name.isupper() for name in names):selected.append(node)
    ns={'__name__':__name__,'__file__':str(P)}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(P),'exec'),ns)
    return ns


def test_mcar_vs_outcome_selection_with_same_expected_loss():
    result=load_definitions()['loss_mechanism_demo']()
    assert result['true_effect']==0
    assert abs(result['rows']['MCAR']['estimate'])<.005
    assert result['rows']['outcome-dependent']['ci'][0]>.05
    assert abs(result['rows']['outcome-dependent']['estimate']-result['selected_limit'])<.005
    fractions=[r['retained']/r['assigned'] for r in result['rows'].values()]
    assert abs(fractions[0]-fractions[1])<.005


def test_mc_interval_does_not_report_zero_risk_after_zero_events():
    ci=load_definitions()['monte_carlo_interval']
    lo,hi=ci(0,250)
    assert lo==0
    assert hi==pytest.approx(1-.025**(1/250))
    assert ci(250,250)==pytest.approx((1-hi,1))


def test_calibration_runs_all_gates_and_does_not_analyze_stopped_data():
    ns=load_definitions();base=ns['ExperimentPlatform']
    reasons=iter(['SRM: fail','ИНВАРИАНТ: fail','GUARDRAIL: fail','план выполнен'])
    seen_seeds=[]
    class Controlled(base):
        def run(self,ticket,**kw):
            seen_seeds.append(kw['rng_seed'])
            assert kw['effect_rel']==0
            assert kw['err_mult']==1.15
            return types.SimpleNamespace(stop_reason=next(reasons))
        def analyze(self,ticket,run):
            assert run.stop_reason=='план выполнен'
            return dict(oec=dict(p=.001,est=.01),guardrail=dict(safe=True))
    with patch.dict(ns,{'ExperimentPlatform':Controlled}):
        result=ns['calibrate_monitoring'](repetitions=4,boundary=True)
    assert len(set(seen_seeds))==4
    assert result['counts']==dict(SRM=1,invariant=1,guardrail=1,any_gate=3,
                                 completed=1,final_oec_reject=1,false_ship=1)
    assert result['rates']['final_oec_reject']==.25  # all runs, not only survivors
    assert result['any_gate_budget']==pytest.approx(.052)


def test_randomized_null_runs_have_nonidentical_counts_and_invariants():
    ns=load_definitions();platform=ns['ExperimentPlatform']()
    doc=ns['DesignDoc']('test','test','интерфейс','конверсия в заказ (2 нед.)',.1)
    run=platform.run(dict(doc=doc,weeks=1,p0=.23),rng_seed=73073)
    assert any(nc!=nt for nc,nt in zip(run.n_c,run.n_t))
    assert np.std(run.z_inv)>0
    # If any data-quality gate fires, the actual path crosses its planned cutoff.
    if run.stop_reason.startswith('SRM'):
        assert run.p_srm[-1]<.001/run.planned_days
    if run.stop_reason.startswith('ИНВАРИАНТ'):
        assert abs(run.z_inv[-1])>ns['stats'].norm.isf(.001/(2*run.planned_days))
