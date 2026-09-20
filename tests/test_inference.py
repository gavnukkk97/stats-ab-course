"""Regression checks for statistical contracts; load pure functions without plotting demos."""
import ast
import math
from pathlib import Path
import unittest
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1] / 'course/modules'


def functions(relative, constants=None):
    tree = ast.parse((ROOT / relative).read_text())
    namespace = dict(np=np, stats=stats, math=math)
    namespace.update(constants or {})
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), namespace)
    return namespace


class InferenceContracts(unittest.TestCase):
    def test_power_null_alpha_and_design(self):
        f = functions('M3/practice/practice_3_1.py', dict(SIGMA=1., ALPHA=.05, P0=.12))
        for alpha in [.01, .05, .1]:
            self.assertAlmostEqual(f['power_z'](0, 100, alpha=alpha), alpha)
        n = f['n_per_group'](.1)
        self.assertAlmostEqual(f['power_z'](.1, n), .8, places=5)
        self.assertAlmostEqual(f['n_one_sample'](.1) * 2, n)
        # Independent exact-binomial simulation is close to the normal approximation.
        simulated = f['simulate_power'](.013, 10000, reps=30000, rng=np.random.default_rng(12))
        theoretical = f['power_z'](.013, 10000, sigma=math.sqrt(.12*.88))
        self.assertLess(abs(simulated - theoretical), .03)

    def test_cuped_matches_ancova_with_within_arm_slope(self):
        f = functions('M3/practice/practice_3_5.py')
        rng = np.random.default_rng(95)
        t = np.repeat([0, 1], 40)
        x = rng.normal(size=80) + 2*t
        y = 3*t + .8*x + rng.normal(size=80)
        result = f['cuped'](x, y, t)
        # Practice returns adjusted effect and theta, checked against direct OLS.
        adjusted = result[0]
        target = np.linalg.lstsq(np.column_stack([np.ones(80), t, x]), y, rcond=None)[0][1]
        self.assertAlmostEqual(adjusted, target)

    def test_oof_predictor_does_not_see_own_outcome(self):
        f = functions('M3/practice/practice_3_6.py', dict(N_PER_ARM=100, TRUE_EFFECT=1.))
        rng = np.random.default_rng(36)
        x = np.column_stack([np.ones(100), rng.normal(size=(100, 3))])
        y = rng.normal(size=100)
        before = f['crossfit_predict'](x, y)
        y[7] += 1000
        after = f['crossfit_predict'](x, y)
        self.assertEqual(before[7], after[7])
        self.assertGreater(np.max(np.abs(before-after)), 1)

    def test_oof_adjustment_keeps_randomized_treatment_effect(self):
        f = functions('M3/practice/practice_3_6.py', dict(N_PER_ARM=100, TRUE_EFFECT=1.))
        rng = np.random.default_rng(360)
        effects, cover = [], []
        for _ in range(250):
            x = np.column_stack([np.ones(500), rng.normal(size=(500, 3))])
            t = np.repeat([0, 1], 250)
            y = 2*t + x @ np.array([0, 2, 3, 4]) + rng.normal(size=500)
            (d, se), _ = f['cupac_estimate'](y, t, x)
            effects.append(d)
            cover.append(abs(d-2) < 1.96*se)
        self.assertLess(abs(np.mean(effects)-2), .03)
        self.assertGreater(np.mean(cover), .91)

    def test_ratio_variance_has_sample_size_and_covariance(self):
        f = functions('M3/practice/practice_3_3.py', dict(B_BOOT=100))
        checks = np.array([10., 80., 50., 12.])
        orders = np.array([1, 4, 2, 1])
        r, var, _, _ = f['ratio_delta_stats'](checks[:, None], orders)
        expected = np.var(checks-r*orders, ddof=1)/(len(checks)*orders.mean()**2)
        self.assertAlmostEqual(var, expected)
        self.assertAlmostEqual(r, checks.sum()/orders.sum())

    def test_sequential_local_horizon_and_martingale_calibration(self):
        f = functions('M3/practice/practice_3_7.py', dict(WORLDS=100, DAYS=60, N_DAY=200, TAU=.1))
        x, z, p = f['simulate_paths'](np.random.default_rng(37), worlds=20000, days=13)
        self.assertEqual(x.shape, (20000, 13))
        v = 2/(200*np.arange(1, 14))
        llr = f['msprt_loglr'](x, v)
        rate = (llr.max(axis=1) >= math.log(20)).mean()
        self.assertLess(rate, .055)
        self.assertGreater((p.min(axis=1) < .05).mean(), .10)
        # Independent density identity, not a copy of the implementation.
        xx = np.array([-.2, 0, .1])
        exact = stats.norm.logpdf(xx, scale=math.sqrt(.02+.01))-stats.norm.logpdf(xx, scale=math.sqrt(.02))
        np.testing.assert_allclose(f['msprt_loglr'](xx, .02), exact)

    def test_rope_includes_harm_and_equivalence(self):
        f = functions('M5/practice/practice_5_3.py', dict(ROPE=.001, EPS=.001))
        self.assertIn('хуже', f['rope_verdict'](np.linspace(-.02, -.01, 1000))[0])
        self.assertIn('лучше', f['rope_verdict'](np.linspace(.01, .02, 1000))[0])
        self.assertIn('эквивалент', f['rope_verdict'](np.linspace(-.0005, .0005, 1000))[0])

    def test_expected_loss_includes_zero_when_choice_correct(self):
        f = functions('M5/practice/practice_5_3.py', dict(ROPE=.001, EPS=.001))
        rng = np.random.default_rng(53)
        a, b = rng.beta(50, 450, 200000), rng.beta(60, 440, 200000)
        grid = f['expected_loss_grid'](50, 450, 60, 440)
        self.assertAlmostEqual(grid, np.maximum(a-b, 0).mean(), delta=.00006)
        self.assertLess(grid, (a-b)[a>b].mean())

    def test_expected_loss_threshold_is_not_uniform_frequentist_risk(self):
        f = functions('M5/practice/practice_5_3.py', dict(ROPE=.001, EPS=.001))
        risk, choose_b, met, _ = f['el_policy_risk'](-.02, (300., 700.), worlds=500)
        self.assertGreater(risk, .01)
        self.assertGreater(met, .99)
        self.assertGreater(choose_b, .9)

    def test_ratio_exact_scale_when_denominator_changes(self):
        xa, ya = np.array([800, 1800, 3000, 2200]), np.array([1, 2, 4, 4])
        xb, yb = np.array([1100, 2000, 1700, 3300]), np.array([1, 2, 2, 4])
        ra, rb = xa.sum()/ya.sum(), xb.sum()/yb.sum()
        signal = (xb-ra*yb).mean()
        self.assertAlmostEqual(signal/yb.mean(), rb-ra)
        self.assertGreater(abs(signal/ya.mean()-(rb-ra)), 30)

    def test_sliding_window_success_plus_failure_equals_window(self):
        f = functions('M5/practice/practice_5_4.py', dict(HORIZON=30, RUNS=4, ARMS=np.array([.1,.2,.3]), EVERY=1))
        class RecordingRNG:
            def __init__(self):
                self.base = np.random.default_rng(540)
                self.calls = 0
            def beta(self, a, b):
                # At every decision, posterior pseudocounts contain precisely the window.
                counts = (a-1)+(b-1)
                np.testing.assert_allclose(counts.sum(axis=1), min(self.calls, 7))
                assert np.min(a) >= 1 and np.min(b) >= 1
                self.calls += 1
                return self.base.beta(a, b)
            def random(self, n):
                return self.base.random(n)
        f['run_policy'](RecordingRNG(), 'window', horizon=30, runs=4, window=7)
        # Separate nonstationary implementation exercises the same invariant.
        ns = functions('M5/practice/practice_5_4.py', dict(HORIZON=30, RUNS=4, ARMS=np.array([.1,.2,.3]), EVERY=1,
                       HOR=30, SWITCH=15, ARMS_NS=np.array([.14,.10])))
        # Names below reflect the explicit nonstationary generator's constants.
        ns.update(ARMS_BEFORE=np.array([.1,.2,.3]), ARMS_AFTER=np.array([.3,.2,.1]))
        ns['run_ns'](RecordingRNG(), 7)


if __name__ == '__main__':
    unittest.main()
