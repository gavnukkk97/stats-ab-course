"""Numerical/statistical regressions for B01–B46 (no figures generated)."""
from pathlib import Path
import importlib.util
import itertools
import unittest
import numpy as np
from scipy import stats

spec = importlib.util.spec_from_file_location('foundations', Path(__file__).resolve().parents[1]/'course/lib/foundations.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


class FoundationsTests(unittest.TestCase):
    def test_coin_exact_two_sided(self):
        np.testing.assert_allclose(f.coin_pvalues()[5:], [1, .75390625, .34375, .109375, .021484375, .001953125])
        np.testing.assert_allclose(f.coin_pvalues(), f.coin_pvalues()[::-1])
        mass = stats.binom.pmf(np.arange(11), 10, .5)
        self.assertAlmostEqual(mass[f.coin_pvalues() <= .05].sum(), .021484375)

    def test_permutation_single_partition_exact_variance(self):
        # Complementary groups must conserve the pooled total, and all balanced
        # partitions have Var(diff)=s_pool²(1/nA+1/nB). Two independent shuffles fail.
        a, b = np.arange(3.), np.arange(3., 6.)
        exact = np.array([np.mean(np.delete(np.r_[a,b], inds))-np.mean(np.r_[a,b][list(inds)])
                          for inds in itertools.combinations(range(6), 3)])
        draws = f.permutation_differences(a,b,np.random.default_rng(14),10000)
        self.assertTrue(set(np.round(draws,10)) <= set(np.round(exact,10)))
        self.assertAlmostEqual(draws.var(), exact.var(), delta=.08)
        self.assertAlmostEqual(exact.var(), np.r_[a,b].var(ddof=1)*2/3)
        self.assertEqual(f.permutation_pvalue(100, draws), 1/10001)

    def test_holm_thresholds(self):
        thresholds=f.holm_thresholds(20)
        self.assertEqual(thresholds[0], .0025)
        self.assertEqual(thresholds[-1], .05)
        self.assertTrue(np.all(np.diff(thresholds)>0))

    def test_two_sided_power(self):
        self.assertAlmostEqual(f.normal_power(0,1), .05)
        np.testing.assert_allclose(f.normal_power(np.arange(-5,6),1), f.normal_power(np.arange(5,-6,-1),1))
        self.assertGreater(f.normal_power(-6,1), .9999)
        self.assertAlmostEqual(f.normal_power(60,300*np.sqrt(2/200)), .5160052739761748)

    def test_type_m_counts_wrong_sign_magnitude(self):
        self.assertEqual(f.type_m(np.array([-6,4]),2),2.5)
        with self.assertRaises(ValueError):f.type_m(np.array([1]),0)

    def test_equal_mean_lognormal_alpha_setup(self):
        for sigma in (1.,2.):
            mu=np.log(1200)-sigma*sigma/2
            self.assertAlmostEqual(np.exp(mu+sigma*sigma/2),1200)

    def test_mw_weak_null_counterexample(self):
        # Both centered normals have theta=.5, but standard MW is not calibrated
        # for this weak null with unequal shapes and unequal sample sizes.
        rng=np.random.default_rng(902)
        a=rng.normal(0,1,(4000,80));b=rng.normal(0,4,(4000,20))
        rejects=stats.mannwhitneyu(a,b,axis=1).pvalue<.05
        self.assertGreater(rejects.mean(), .10)

    def test_mw_small_n_and_ties(self):
        a=[1,2,3,5];b=[4,6,7,8,9]
        result=stats.mannwhitneyu(b,a,method='exact')
        self.assertEqual(result.statistic,19)
        self.assertAlmostEqual(result.pvalue, .031746031746031744)
        a=[400,500,600];b=[400,500,600,20000]
        result=stats.mannwhitneyu(b,a,method='asymptotic')
        self.assertEqual(result.statistic,7.5)
        self.assertAlmostEqual(result.pvalue,.7162897505408428)
        self.assertEqual(np.median(b),550)

    def test_bootstrap_median_support(self):
        x=np.array([2,4,4,6,8,10,10,12])
        support={(a+b)/2 for a,b in itertools.combinations_with_replacement(x,2)}
        self.assertEqual(support,set(range(2,13)))

    def test_anova_homework(self):
        F=(150/3)/(360/28)
        self.assertAlmostEqual(F,3.888888888888889)
        self.assertAlmostEqual(stats.f.sf(F,3,28),.019266023685701798,places=7)
        self.assertAlmostEqual(stats.f.ppf(.95,3,28),2.9466852660172655,places=7)

    def test_whale_arithmetic_and_lognormal_truth(self):
        orders=np.array([690,720,760,810,850,890,980,1240,1580,11200])
        self.assertEqual(np.median(orders),870)
        self.assertEqual(np.median(orders[:-1]),850)
        self.assertEqual(np.mean([690,720,760,810,500000]),100596)
        self.assertAlmostEqual(np.exp(np.log(900)+1.1**2/2),1648.127,places=2)
        self.assertEqual((11200+8000+72000)/100,912)

    def test_bh_step_up_counterexample(self):
        p=np.r_[np.full(300,.008),np.full(1500,.5)]
        adjusted=stats.false_discovery_control(p,method='bh')
        self.assertEqual(np.count_nonzero(adjusted<=.05),300)
        np.testing.assert_allclose(adjusted[:300],.048)

    def test_sign_and_t_critical(self):
        self.assertAlmostEqual(stats.binomtest(7,8).pvalue,.0703125)
        self.assertAlmostEqual(stats.t.ppf(.975,3),3.182446305284263)

    def test_observed_ci_does_not_exclude_three_pp(self):
        se=np.sqrt(.129*.871*(1/2000+1/2000))
        low,high=.018+np.array([-1,1])*stats.norm.ppf(.975)*se
        self.assertLess(low,0)
        self.assertGreater(high,.035)

    def test_report_mean_ci(self):
        se=300*np.sqrt(2/2000)
        low,high=40+np.array([-1,1])*stats.t.ppf(.975,3998)*se
        self.assertAlmostEqual(se,9.486832980505138)
        self.assertAlmostEqual(low,21.4,delta=.02)
        self.assertAlmostEqual(high,58.6,delta=.02)

    def test_mc_interval_not_point_estimate(self):
        low,high=f.mc_interval(75,1500)
        self.assertLess(low,.05)
        self.assertGreater(high,.05)
        self.assertGreater(high-low,.02)
        self.assertGreater(f.mc_interval(0,100)[1],0)

    def test_retests_binary_model(self):
        # Repeated evidence about one fixed hypothesis differs from three
        # separately selected hypotheses. Reference values from case 12.
        pi, power, alpha = .2, .8, .05
        ppv = pi*power/(pi*power+(1-pi)*alpha)
        triple = pi*power**3/(pi*power**3+(1-pi)*alpha**3)
        mixed = pi*power*(1-power)/(pi*power*(1-power)+(1-pi)*alpha*(1-alpha))
        self.assertAlmostEqual(ppv, .8)
        self.assertAlmostEqual(triple, .9990243902439024)
        self.assertAlmostEqual(ppv**3, .512)
        self.assertAlmostEqual(mixed, .4571428571428572)

    def test_replication_reference_comparison(self):
        # Case 16: threshold crossing changes, estimates remain compatible.
        self.assertLess(2*stats.norm.sf(1/.4), .05)
        self.assertGreater(2*stats.norm.sf(.6/.4), .05)
        p_difference=2*stats.norm.sf((1-.6)/np.sqrt(.4**2+.4**2))
        self.assertAlmostEqual(p_difference,.4795001221869535)


if __name__ == '__main__':unittest.main()
