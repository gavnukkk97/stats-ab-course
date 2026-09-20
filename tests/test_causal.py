"""Regression checks for the causal examples' estimands and inference rules."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from scipy import stats

P = Path(__file__).resolve().parents[1] / 'course/lib/causal_utils.py'
spec = importlib.util.spec_from_file_location('causal_utils', P)
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


class CausalChecks(unittest.TestCase):
    def test_collider_descendant_opens_path(self):
        edges=[('X','S'),('Y','S'),('S','Z')]
        self.assertTrue(c.path_blocked(edges,['X','S','Y'],set()))
        self.assertFalse(c.path_blocked(edges,['X','S','Y'],{'Z'}))
        self.assertTrue(c.path_blocked([('Z','X'),('Z','Y')],['X','Z','Y'],{'Z'}))

    def test_rosenbaum_bound_monotonic_and_returns_to_nonsignificance(self):
        gamma=np.linspace(1,5,161)
        p=np.array([c.rosenbaum_upper_p(909,1214,g) for g in gamma])
        self.assertTrue(np.all(np.diff(p)>=-1e-14))
        self.assertLess(p[0],.05);self.assertGreater(p[-1],.95)
        self.assertAlmostEqual(c.rosenbaum_upper_p(8,10,1),stats.binom.sf(7,10,.5))

    def test_evalue_null_crossing_and_protective_boundary(self):
        self.assertEqual(c.evalue_interval(1.2,.8,1.5)[1],1)
        self.assertAlmostEqual(c.evalue_interval(.5,.4,.7)[1],c.evalue_point(.7))
        self.assertAlmostEqual(c.evalue_point(.5),c.evalue_point(2))
        # E is the equal-strength solution, not a minimum on each unequal association.
        a,b=4,1.5;self.assertGreaterEqual(a*b/(a+b-1),1.3)
        self.assertLess(b,c.evalue_point(1.3))

    def test_paired_rr_accounts_for_covariance(self):
        t=np.array([1]*40+[0]*60);r,lo,hi=c.paired_rr_interval(t,t)
        self.assertEqual((r,lo,hi),(1,1,1))
        control=np.roll(t,20);_,lo2,hi2=c.paired_rr_interval(t,control)
        self.assertLess(lo2,1);self.assertGreater(hi2,1)

    def test_exact_pairs_disjoint_and_match_only_retained_treated(self):
        x=np.array([[0],[0],[1],[1],[10],[2]])
        d=np.array([1,0,1,0,1,0])
        it,ic=c.exact_pairs(x,d)
        np.testing.assert_array_equal(x[it],x[ic])
        self.assertEqual(len(set(it)|set(ic)),2*len(it))
        self.assertEqual(c.matched_smd(x[:,0],d,it,ic),0)
        self.assertGreater(x[d==1].mean()-x[ic].mean(),0)

    def test_stabilization_does_not_change_hajek_or_ess(self):
        w=np.array([1.,2.,8.,15.]);y=np.array([1.,3.,2.,9.]);stabilized=.37*w
        self.assertAlmostEqual(np.average(y,weights=w),np.average(y,weights=stabilized))
        self.assertAlmostEqual(c.effective_size(w),c.effective_size(stabilized))

    def test_no_synthetic_control_without_donors_and_no_universal_rank(self):
        self.assertIn('не обоснована',c.method_options(single_unit=True,has_preperiod=True)[0])
        opts=c.method_options(has_threshold=True,has_instrument=True,has_control_group=True,
                              parallel_trends=True,has_preperiod=True)
        self.assertEqual(len(opts),3)
        self.assertTrue(any('Synthetic' in x for x in c.method_options(single_unit=True,has_preperiod=True,has_donors=True)))

    def test_plr_and_aipw_target_different_heterogeneous_estimands(self):
        res=c.crossfit_binary_demo()
        self.assertGreater(res['overlap_truth']-res['ate'],1.7)
        self.assertLess(abs(res['plr']-res['overlap_truth']),.15)
        self.assertLess(abs(res['aipw']-res['ate']),.15)

    def test_cluster_interval_coverage_with_shared_city_shocks(self):
        # Repeated samples: users within cities do not create independent assignments.
        rng=np.random.default_rng(616);g,m=40,8
        clusters=np.repeat(np.arange(g),m);d=np.repeat(np.arange(g)>=g//2,m).astype(float)
        X=np.column_stack([np.ones(g*m),d]);covered,naive_covered=0,0
        for _ in range(200):
            y=2*d+np.repeat(rng.normal(0,3,g),m)+rng.normal(size=g*m)
            beta,cov=c.cluster_covariance(y,X,clusters)
            covered+=abs(beta[1]-2)<=stats.t.ppf(.975,g-1)*np.sqrt(cov[1,1])
            residual=y-X@beta
            naive=(residual@residual)/(g*m-2)*np.linalg.inv(X.T@X)
            naive_covered+=abs(beta[1]-2)<=1.96*np.sqrt(naive[1,1])
        self.assertGreater(covered,175);self.assertLess(naive_covered,145)


if __name__=='__main__': unittest.main()
