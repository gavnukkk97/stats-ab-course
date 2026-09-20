"""Регрессии смысла: план/решение, ratio CI, окна событий и ID назначения."""
import ast
import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def definitions(relative):
    tree = ast.parse((ROOT / relative).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    ns = dict(np=np, pd=pd, stats=stats, math=math, copy=copy, hashlib=hashlib,
              ALPHA=.05, POWER=.8, WEEKDAY=1500, WEEKEND=2100, N_BUCKETS=1000,
              LAMBDA_PAIR=.005)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, 'exec'), ns)
    return ns


def test_detection_power_is_not_business_rule_power():
    ns = definitions('course/modules/M4/practice/practice_4_1.py')
    n = math.ceil(ns['n_per_group'](.015, math.sqrt(.23 * .77)) * 1.05)
    detect, practical = ns['oec_rule_power'](n, .23, .015, .005)
    assert .79 < detect < .82
    assert .46 < practical < .49
    assert ns['oec_rule_power'](n, .23, .015, .01)[1] < practical
    # Повторные оценки проверяют вероятность прохождения именно нижней CI-границы.
    se = math.sqrt((.23 * .77 + .245 * .755) / n)
    estimates = np.random.default_rng(10).normal(.015, se, 100_000)
    assert np.mean(estimates - stats.norm.isf(.025) * se > .005) == pytest.approx(practical, abs=.006)


def test_analysis_waits_for_last_enrolled_window():
    build = definitions('course/modules/M4/practice/practice_4_1.py')['build_design_doc']
    doc, plan = build('test', 'test', {'x': 'x', 'y': 'y', 'z': 'z'}, 'CR', .23, .015, [])
    assert plan['enrollment_end'] - plan['t0'] == pd.Timedelta(weeks=plan['weeks'])
    assert plan['analysis_date'] - plan['enrollment_end'] == pd.Timedelta(days=14)
    assert plan['expected_n'] >= plan['n']
    assert plan['calendar_practical_power'] > plan['practical_power']
    assert str(plan['analysis_date'].date()) in doc
    assert 'только OEC' in doc


def test_bootstrap_relative_interval_resamples_denominator():
    fn = definitions('course/modules/M4/practice/practice_4_4.py')['bootstrap_mean_contrasts']
    # Контрольные средние после ресемплинга: 1,2,3. Отношения: 5,2,1.
    absolute, relative = fn(np.array([6., 6.]), np.array([1., 3.]), np.random.default_rng(12), 2000)
    np.testing.assert_allclose(absolute, [3, 5])
    np.testing.assert_allclose(relative, [1, 5])
    assert not np.allclose(relative, absolute / 2)
    _, scaled = fn(np.array([600., 600.]), np.array([100., 300.]), np.random.default_rng(12), 2000)
    np.testing.assert_allclose(scaled, relative)


def test_relative_delta_uses_control_derivative():
    fn = definitions('course/modules/M4/practice/practice_4_5.py')['relative_lift_and_se']
    lift, se = fn(200, 100, 100, 100, 400, 20)
    gradient = np.array([1/100, -200/100**2])
    covariance = np.diag([100/100, 400/20])
    assert lift == 1
    assert se**2 == pytest.approx(gradient @ covariance @ gradient)
    assert se == pytest.approx(.09)
    assert fn(2000, 10000, 100, 1000, 40000, 20) == pytest.approx((lift, se))


def test_segment_relative_interval_uses_control_derivative():
    fn = definitions('course/modules/M4/practice/practice_4_4.py')['relative_mean_ci']
    treatment = np.array([18., 20., 22.]); control = np.array([5., 10., 15.])
    lift, ci = fn(treatment, control)
    se = math.sqrt(4/3 + 4*25/3) / 10
    assert lift == 1
    np.testing.assert_allclose(ci, [1 - 1.96*se, 1 + 1.96*se])


@pytest.mark.parametrize('duplicate_layer', ['rank', 'ui'])
def test_duplicate_experiment_id_is_rejected_without_mutation(duplicate_layer):
    cls = definitions('course/modules/M7/practice/practice_7_1.py')['SplitService']
    svc = cls(); svc.add_domain('all'); svc.add_layer('rank', 'all'); svc.add_layer('ui', 'all')
    svc.add_experiment('rank', 'same-id', .1)
    before = svc.snapshot()
    with pytest.raises(ValueError, match='уникальным'):
        svc.add_experiment(duplicate_layer, 'same-id', .1)
    assert svc.snapshot() == before


def test_session_window_matches_post_exposure_event_window():
    fn = definitions('course/modules/M7/practice/practice_7_2.py')['post_exposure_day_fractions']
    result = fn(np.array([0, 12, 24, 30, 72]), 3)
    np.testing.assert_allclose(result, [[1,1,1], [.5,1,1], [0,1,1], [0,.75,1], [0,0,0]])
    np.testing.assert_allclose(result.sum(axis=1), np.array([72,60,48,42,0])/24)
