"""Candidate availability must not be mistaken for an observational effect."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/post-processing'))
from observation_convergence_utils import parameter_reference
from model_analysis_utils import eligible_pool_candidates,available_score_pools,analyse_pool_views


class PoolViewTests(unittest.TestCase):
    def inputs(self):
        data=pd.DataFrame(dict(evaluation_id=list('abcdefg'),history=['h']*7,
            usable=[True]*7,cost_terms_match=[True]*5+[False,True],
            known_physical_metadata=[True]*7,cost=[1,2,3,4,5,6,7],
            budget=[3]*4+[1,3,3],configured_max_budget=[3]*7,
            parameter_hash=list('abcdafg'),score_sed=[1,2,3,4,5,6,np.inf],
            score_alma=[np.nan,2,1,1,1,1,1],param__x=np.linspace(0,1,7)))
        inventory=pd.DataFrame([dict(history='h',configspace_hash='space',template_hash='template',photon_settings='photons')])
        records={name:dict(scoring_metadata=dict(numerical_fidelity={},products=['sed','alma'],
            interferometry={},pdi={},alma_observed={'energy':1 if name!='d' else 2})) for name in data.evaluation_id}
        return data,inventory,records

    def test_native_audit_and_final_budget_exclusions(self):
        data,inventory,records=self.inputs()
        result=eligible_pool_candidates(data,inventory,records)
        self.assertEqual(set(result.evaluation_id),set('abcdg'))
        self.assertEqual(result.physical_pool_key.nunique(),1)

    def test_only_relevant_scoring_metadata_partitions(self):
        data,inventory,records=self.inputs()
        candidates=eligible_pool_candidates(data,inventory,records)
        sed=available_score_pools(candidates,records,['SED'])
        self.assertEqual(len(sed),1)
        self.assertEqual(set(next(iter(sed.values())).evaluation_id),set('abcd'))
        both=available_score_pools(candidates,records,['SED','ALMA'])
        self.assertEqual(len(both),2) # different ALMA definitions must not mix
        self.assertEqual(set(pd.concat(both.values()).evaluation_id),set('bcd'))

    def test_available_winner_differs_from_shared_winner(self):
        data,inventory,records=self.inputs()
        original=data.copy(deep=True)
        candidates=eligible_pool_candidates(data,inventory,records)
        tables=analyse_pool_views(candidates,records,{'SED':['SED'],'SED+ALMA':['SED','ALMA']},
            {'x':{'name':'x','type':'uniform_float','lower':0.,'upper':1.}},
            {'x':parameter_reference(candidates,'x',{'type':'uniform_float','lower':0.,'upper':1.})},
            {'allowance':[1.]},min_models=1)
        winners=tables['winners']
        available=winners[(winners.view=='available_per_objective')&(winners.selected_by=='SED')]
        self.assertEqual(available.evaluation_id.tolist(),['a'])
        shared=winners[(winners.view=='shared_per_comparison')&(winners.selected_by=='SED')]
        self.assertEqual(set(shared.evaluation_id),set('bd'))
        self.assertTrue(tables['effects'].view.eq('shared_per_comparison').all())
        counts=tables['counts'].query("view == 'shared_per_comparison'")
        self.assertTrue(counts.groupby(['score_pool_key','scope']).n_pool.nunique().eq(1).all())
        pd.testing.assert_frame_equal(data,original)

    def test_empty_objective_is_reported_not_filled(self):
        data,inventory,records=self.inputs()
        candidates=eligible_pool_candidates(data,inventory,records)
        tables=analyse_pool_views(candidates,records,{'INT':['INT']},{},{},{'allowance':[.1]})
        self.assertEqual(tables['availability'].n_eligible.tolist(),[0])
        self.assertTrue(tables['winners'].empty)

if __name__=='__main__':unittest.main()
