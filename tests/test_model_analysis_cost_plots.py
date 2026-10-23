"""Verify plotted cost selections and parameter coverage."""
import sys
import unittest
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/post-processing'))
from model_analysis_utils import plot_available_pool_costs
from observation_convergence_utils import parameter_reference

class CostPlotTests(unittest.TestCase):
    def test_thresholds_categories_missing_and_pagination(self):
        pool=pd.DataFrame({'evaluation_id':list('abcdef'),'parameter_hash':list('aabcde'),
            'history':['h']*6,'score_pool_key':['same']*6,
            'score_sed':[10,11,12,15,20,21],
            'param__x':[1,2,3,4,5,np.nan], 'param__category':['a','b']*3})
        refs={'x':parameter_reference(pool,'x',{'type':'uniform_float','lower':1,'upper':5}),
              'category':parameter_reference(pool,'category',{'type':'categorical','choices':['a','b']})}
        original=pool.copy(deep=True)
        figs,counts,coverage=plot_available_pool_costs(pool,'SED',['SED'],refs,panels_per_page=1)
        self.assertEqual(len(figs),2)
        self.assertEqual(counts.cutoff.tolist(),[20,15,11])
        self.assertEqual(counts.n_selected.tolist(),[5,4,2])
        self.assertEqual(coverage.n_valid.tolist(),[5,6])
        self.assertEqual(coverage.n_missing_or_invalid.tolist(),[1,0])
        for fig in figs:fig.canvas.draw();plt.close(fig)
        pd.testing.assert_frame_equal(pool,original)

    def test_zero_and_negative_best(self):
        for costs,cutoffs in [([0,1,2],[0,0,0]),([-10,-5,1],[0,-5,-9])]:
            pool=pd.DataFrame({'evaluation_id':list('abc'),'parameter_hash':list('abc'),
                'history':['h']*3,'score_pool_key':['same']*3,'score_sed':costs,'param__x':[1,2,3]})
            refs={'x':parameter_reference(pool,'x',{'lower':1,'upper':3})}
            figs,counts,_=plot_available_pool_costs(pool,'SED',['SED'],refs)
            self.assertEqual(counts.cutoff.tolist(),cutoffs)
            self.assertEqual(figs[0].axes[0].get_yscale(),'linear')
            for fig in figs:fig.canvas.draw();plt.close(fig)

if __name__=='__main__':unittest.main()
