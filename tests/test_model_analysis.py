"""Scientific invariants for read-only saved-model comparisons."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/post-processing'))
from model_analysis_utils import (score_matrix,objective_design,addition_design,
    competitive_mask,distribution_effect,mixed_distance,alternative_families,
    pareto_mask,bootstrap_history_effects)
from run_summary_utils import SUBSETS


class ModelAnalysisTests(unittest.TestCase):
    def pool(self):
        pool=pd.DataFrame({'evaluation_id':['a','b','c','d'], 'history':['h']*4,
            'param__x':[0.,.1,.9,1.], 'param__category':['a','a','b','b']})
        for column in SUBSETS['All']:pool[column]=0.
        pool['score_sed']=[1.,1.,2.,3.]
        pool['score_alma']=[3.,2.,1.,1.]
        return pool

    def references(self):
        return {'x':{'kind':'linear','lower':0.,'upper':1.},
                'category':{'kind':'categorical','categories':['a','b']}}

    def test_original_score_sum_unique_terms_and_no_mutation(self):
        pool=self.pool();original=pool.copy(deep=True)
        result=score_matrix(pool,{'overlap':['SED','SED','ALMA']})
        np.testing.assert_allclose(result.overlap,[4,3,3,4])
        pd.testing.assert_frame_equal(pool,original)
        pool.loc[0,'score_alma']=np.nan
        with self.assertRaises(ValueError):score_matrix(pool,{'all':['SED','ALMA']})

    def test_selection_ties_zero_and_negative_optimum(self):
        mask,cut=competitive_mask(pd.Series([1.,1.,2.,3.]),'fraction',.25)
        self.assertEqual(mask.sum(),2)
        mask,cut=competitive_mask(pd.Series([0.,.1,.2]),'allowance',.5,.1)
        self.assertEqual(mask.sum(),2)
        _,cut=competitive_mask(pd.Series([-10.,-8.,0.]),'allowance',.2)
        self.assertEqual(cut,-8.)

    def test_factorial_and_conditional_additions(self):
        design=objective_design();additions=addition_design(design)
        self.assertEqual(sum('SED' in values for values in design.values()),8)
        self.assertTrue(((additions.before=='SED+PDI+ALMA')&(additions.after=='All')
                         &(additions.added=='INT')).any())
        self.assertTrue(((additions.before=='INT+PDI+ALMA')&(additions.added=='SED')).any())

    def test_distribution_change_distinct_from_narrowing(self):
        pool=self.pool();ref=self.references()
        effect=distribution_effect(pool.iloc[:2],pool.iloc[2:],'x',ref['x'])
        self.assertAlmostEqual(effect['shift'],.9)
        self.assertAlmostEqual(effect['width_ratio'],1.)
        cat=distribution_effect(pool.iloc[:2],pool.iloc[2:],'category',ref['category'])
        self.assertEqual(cat['shift'],1.)
        self.assertTrue(np.isnan(cat['width_ratio']))

    def test_distinct_equal_score_families_keep_every_model(self):
        pool=self.pool();ref=self.references()
        distance=mixed_distance(pool,ref)
        self.assertAlmostEqual(distance[0,1],.05)
        self.assertEqual(distance[0,3],1.)
        members,summary=alternative_families(pool,pd.Series([1.,1.,1.,1.]),ref,
                                            allowance=0.,distance_cuts=[.1])
        self.assertEqual(summary.shape[0],2)
        self.assertEqual(members.evaluation_id.nunique(),4)
        self.assertTrue(members.score_excess.eq(0.).all())

    def test_pareto_retains_ties_and_rejects_dominated_points(self):
        np.testing.assert_array_equal(pareto_mask([[1,3],[2,2],[3,1],[2,3],[1,3]]),
                                      [True,True,True,False,True])

    def test_no_bootstrap_with_two_histories(self):
        with self.assertRaises(ValueError):bootstrap_history_effects(pd.DataFrame(),['h1','h2'])

if __name__=='__main__':unittest.main()
