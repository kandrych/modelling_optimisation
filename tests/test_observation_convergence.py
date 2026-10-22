"""Checks for same-pool observational restriction diagnostics."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/post-processing'))
from observation_convergence_utils import model_pool,select_models,restriction_analysis,parameter_metrics
from run_summary_utils import SUBSETS

class RestrictionTests(unittest.TestCase):
    def pool(self):
        p=pd.DataFrame({'parameter_hash':list('abcde'),'param__x':[1.,2.,3.,4.,5.]})
        for c in SUBSETS['All']:p[c]=0.
        p['score_sed']=[1.,2.,3.,4.,5.]
        p['score_alma']=[20.,10.,0.,10.,20.]
        return p

    def test_physical_quantiles_and_tied_best_values(self):
        pool=self.pool()
        pool['param__x']=[.001,.01,.1,1.,10.]
        pool['score_sed']=[1.,1.,2.,3.,4.]
        definition={'type':'uniform_float','lower':.001,'upper':10.,'log':True}
        result,_,_=parameter_metrics(pool,pool.iloc[:2],'x',definition,min_models=2)
        self.assertAlmostEqual(result['median_in_parameter_units'],.0055)
        self.assertAlmostEqual(result['q25_in_parameter_units'],.00325)
        self.assertAlmostEqual(result['width'],.8)
        metrics,_,_=restriction_analysis(pool,{'x':definition},['SED','SED+ALMA'],
                                         ['x'],[0.],[],min_models=2)
        import json
        baseline=metrics.loc[metrics.subset.eq('SED')].iloc[0]
        self.assertEqual(json.loads(baseline.best_model_parameter_values),[.001,.01])
        self.assertEqual(baseline.minimum_in_parameter_units,.001)

    def test_sequences_follow_actual_products_and_keep_provenance(self):
        from observation_convergence_utils import fidelity_sequences
        f=pd.DataFrame([dict(evaluation_id=str(i),history=h,budget=b,stage=str(b),usable=True)
            for i,(h,b) in enumerate([('staged',1),('staged',2),('staged',3),('alma_only',1)])])
        products=[['sed'],['sed','vis2_chromatic'],
                  ['sed','vis2_chromatic','pdi_V','pdi_I','pdi_H','alma'],['sed','alma']]
        records={str(i):{'info':{'fidelity':{'products':p}}} for i,p in enumerate(products)}
        paths,stages,provenance=fidelity_sequences(f,records)
        self.assertEqual(len(paths),1)
        path=next(iter(paths.values()))
        self.assertEqual(path[1],'SED+INT(chrom)')
        self.assertNotIn('ALMA_first',paths)
        self.assertEqual(provenance.iloc[0].histories,'staged')
        comparisons,_,_=fidelity_sequences(f,records,'observed_products')
        self.assertTrue(any('SED+ALMA' in path for path in comparisons.values()))
        single=f[f.history=='alma_only']
        self.assertFalse(fidelity_sequences(single,records)[0])

    def test_pool_scores_preserve_missing_terms_and_parameter_filters(self):
        from observation_convergence_utils import pool_score_summary
        p=self.pool()
        p['usable']=True;p['known_physical_metadata']=True;p['comparison_key']='compatible'
        p['objective_key']='All';p['budget']=1.;p['products']='all';p['stage']='all';p['history']='first'
        p['cost']=p[SUBSETS['All']].sum(axis=1)
        repeat=p.iloc[:1].copy();repeat['score_sed']=3.;repeat['cost']+=2
        p=pd.concat([p,repeat],ignore_index=True)
        p['evaluation_id']=[f'evaluation_{i}' for i in range(len(p))]
        summary,_=pool_score_summary(p,['SED','All'],{'x':{'min':1,'max':2}})
        row=summary[summary.parameter_hash=='a'].iloc[0]
        self.assertEqual(len(summary),3)
        self.assertEqual(summary.model_id.nunique(),3)
        self.assertEqual(summary.loc[summary.parameter_hash=='a','score_sed'].tolist(),[1.,3.])
        self.assertEqual(row.selected_total,row.cost)
        p.loc[p.parameter_hash=='b','score_alma']=np.nan
        summary,_=pool_score_summary(p,['SED','ALMA'],parameter_hashes=['b'])
        self.assertFalse(summary.iloc[0].selected_total_complete)
        self.assertTrue(np.isnan(summary.iloc[0].selected_total))
        self.assertTrue(np.isnan(summary.iloc[0].score_alma))
        with self.assertRaises(ValueError):pool_score_summary(p,['nonexistent'])

    def test_convergence_views_preserve_rows_and_separate_incompatible_repeats(self):
        from observation_convergence_utils import convergence_views
        p=self.pool().iloc[:1].copy()
        p['comparison_key']='same';p['known_physical_metadata']=True;p['budget']=1.
        p['instance']='star';p['history']='first';p['seed']=1
        repeat=p.copy();repeat['score_sed']=3.;repeat['seed']=2
        other=p.copy();other['comparison_key']='different'
        budget=p.copy();budget['budget']=2.
        instance=p.copy();instance['instance']='another star'
        unknown=p.copy();unknown['known_physical_metadata']=False
        rows=pd.concat([p,repeat,other,budget,instance,unknown,unknown],ignore_index=True)
        rows['evaluation_id']=[f'row_{i}' for i in range(len(rows))]
        rows['model_id']=[f'Model {i}' for i in range(len(rows))]
        views,scatter=convergence_views(rows)
        self.assertEqual(len(views['evaluation']),7)
        self.assertEqual(len(views['configuration']),6)
        averaged=views['configuration'].loc[lambda f:f.realisations.eq(2)].iloc[0]
        self.assertEqual(averaged.score_sed,2.)
        self.assertNotIn('model_id',views['configuration'])
        self.assertEqual(views['evaluation'].score_sed.tolist(),rows.score_sed.tolist())
        variability=scatter[(scatter.configuration_id==averaged.configuration_id)&(scatter.score=='score_sed')].iloc[0]
        self.assertAlmostEqual(variability['std'],np.sqrt(2.))
        self.assertEqual(variability.n_simulation_seeds,2)

    def test_shared_reference_keeps_histogram_bins_and_angles_fixed(self):
        from observation_convergence_utils import parameter_reference
        pool=self.pool();pool['param__x']=[179.,0.,1.,178.,2.]
        ref=parameter_reference(pool,'x',{},period=180,bins=8)
        a,h,edges=parameter_metrics(pool,pool.iloc[:3],'x',{},min_models=2,period=180,reference=ref)
        b,h2,edges2=parameter_metrics(pool.iloc[:3],pool.iloc[:3],'x',{},min_models=2,period=180,reference=ref)
        self.assertEqual(a['circular_origin'],b['circular_origin'])
        np.testing.assert_allclose(h,h2)
        np.testing.assert_allclose(edges,edges2)
        self.assertAlmostEqual(a['width'],b['width'])

    def test_invalid_numeric_values_do_not_become_categories_or_support(self):
        from observation_convergence_utils import parameter_reference
        pool=self.pool();pool['param__x']=[1.,10.,0.,float('inf'),'bad']
        definition={'type':'uniform_float','log':True,'lower':1,'upper':100}
        ref=parameter_reference(pool,'x',definition)
        row,h,_=parameter_metrics(pool,pool,'x',definition,min_models=3,reference=ref)
        self.assertEqual(row['kind'],'log10')
        self.assertEqual(row['n_selected'],2)
        self.assertEqual(row['n_invalid'],3)
        self.assertFalse(row['supported'])
        self.assertAlmostEqual(h.sum(),1.)

    def test_marginal_bars_and_histograms_use_declared_reference(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from observation_convergence_utils import parameter_reference,plot_parameter_marginals
        pool=self.pool();pool['param__category']=['a','b','a','b','a']
        refs={'x':parameter_reference(pool,'x',{'lower':0,'upper':10},bins=5),
              'category':parameter_reference(pool,'category',{'type':'categorical','choices':['a','b','c']})}
        row,h,edges=parameter_metrics(pool,pool,'x',{},reference=refs['x'])
        fig=plot_parameter_marginals(pool,{'SED':pool},['x','category'],refs)
        self.assertEqual(len(fig.axes[1].patches),3)  # Includes unvisited declared category.
        heights=[patch.get_height() for patch in fig.axes[1].patches]
        np.testing.assert_allclose(heights,[.6,.4,0.])
        np.testing.assert_allclose(edges,[0,2,4,6,8,10])
        self.assertIn('histogram',fig.axes[0].get_title())
        self.assertIn('category',fig.axes[1].get_title())
        plt.close(fig)

    def test_sequence_products_cannot_relabel_another_interferometry_mode(self):
        from observation_convergence_utils import compatible_sequence_paths
        stages=pd.DataFrame([{'objective':'SED','products':'sed'},
            {'objective':'SED+INT(chrom)','products':'sed, vis2_chromatic'}])
        selected=pd.DataFrame({'evaluation_id':['one']})
        records={'one':{'info':{'fidelity':{'products':['sed','vis2_1perband']}}}}
        sequences,audit=compatible_sequence_paths({'path':['SED','SED+INT(chrom)']},stages,selected,records)
        self.assertFalse(sequences)
        self.assertFalse(audit.iloc[0].compatible)
        self.assertEqual(audit.iloc[0].missing_products,'vis2_chromatic')

    def test_stage_discovery_has_no_global_side_effect_and_handles_regression(self):
        from observation_convergence_utils import fidelity_sequences,objective_columns
        before={k:list(v) for k,v in SUBSETS.items()}
        products=[['sed'],['sed','alma'],['sed'],['sed','pdi_I']]
        f=pd.DataFrame([dict(evaluation_id=str(i),history='h',budget=i,stage=str(i),usable=True)
                        for i in range(4)])
        records={str(i):{'info':{'fidelity':{'products':p}}} for i,p in enumerate(products)}
        paths,_,_=fidelity_sequences(f,records)
        self.assertEqual(set(paths),{'SED → SED+ALMA','SED → SED+PDI(I)'})
        self.assertEqual(SUBSETS,before)
        self.assertEqual(objective_columns('SED+PDI(I)'),['score_sed','score_pdi_I'])

    def test_parameter_score_panels_keep_added_scores_and_selection_distinct(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from observation_convergence_utils import parameter_reference,parameter_score_diagnostic
        pool=self.pool()
        ref=parameter_reference(pool,'x',{'lower':1,'upper':5},bins=2)
        fig,audit,bins=parameter_score_diagnostic(pool,'x',ref,['SED','SED+ALMA'],
            ['SED','ALMA'],allowance=0,min_bin_models=2)
        added=audit[audit.kind=='added'].iloc[0]
        ax=fig.axes[int(added.panel)]
        np.testing.assert_allclose(ax.collections[0].get_offsets()[:,1],pool.score_alma)
        # The new total wins at x=3, although SED alone wins at x=1.
        np.testing.assert_allclose(ax.collections[1].get_offsets()[:,0],[3.])
        self.assertEqual(added.selection_objective,'SED+ALMA')
        self.assertEqual(added.n_competitive,1)
        self.assertTrue(bins.groupby('panel').n.sum().eq(len(pool)).all())
        ecdf=fig.axes[len(audit)]
        self.assertEqual(ecdf.lines[0].get_ydata()[-1],1.)
        self.assertEqual(len(ecdf.lines),3)
        plt.close(fig)

    def test_parameter_score_categories_and_nonpositive_log_scores_remain_visible(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from observation_convergence_utils import parameter_reference,parameter_score_diagnostic
        pool=self.pool();pool['param__category']=['a','a','b','b','b']
        ref=parameter_reference(pool,'category',{'type':'categorical','choices':['a','b','c']})
        fig,audit,bins=parameter_score_diagnostic(pool,'category',ref,['SED','SED+ALMA'],
            ['PDI'],min_bin_models=2,score_scale='log')
        self.assertEqual(audit.iloc[0].score_scale,'linear')
        self.assertEqual(len(fig.axes[0].collections[0].get_offsets()),5)
        self.assertEqual(len(fig.axes[len(audit)].patches),6)
        self.assertFalse(bins[bins.bin_or_category=='c'].supported.any())
        plt.close(fig)

    def test_repeat_vectors_remain_unique_models_without_averaging(self):
        p=self.pool();p=pd.concat([p,p.iloc[:1].assign(score_sed=3.)])
        r=model_pool(p)
        self.assertEqual(r.loc[r.parameter_hash=='a','score_sed'].tolist(),[1.,3.])
        self.assertTrue(r.realisations.eq(1).all())
        self.assertEqual(len(r),6)

    def test_matched_fraction_includes_ties_and_absolute_floor(self):
        p=self.pool();p['score_sed']=[1,1,1,2,3]
        s,_=select_models(p,'SED','fraction',.2)
        self.assertEqual(len(s),3)
        s,cutoff=select_models(p,'SED','allowance',0.,1.)
        self.assertEqual(cutoff,2.)
        self.assertEqual(len(s),4)

    def test_addition_restriction_and_threshold_support(self):
        m,c,d=restriction_analysis(self.pool(),{'x':{'lower':1,'upper':5}},
            ['SED','SED+ALMA'],['x'],[1.], [.4], min_models=2)
        row=d[d['mode']=='allowance'].iloc[0]
        self.assertFalse(row.supported)
        self.assertEqual(row.width_ratio,0.)
        self.assertEqual(c.iloc[0].n_pool,5)
        with self.assertRaises(ValueError):
            restriction_analysis(self.pool(),{},['SED+ALMA','SED'],['x'],[1.],[])

    def test_categories_logarithms_and_circular_wrapping(self):
        pool=self.pool();pool['param__x']=[1.,10.,100.,1000.,10000.]
        row,_,_=parameter_metrics(pool,pool,'x',{'log':True,'lower':1.,'upper':10000.},2)
        self.assertAlmostEqual(row['median'],2.)
        row,h,_=parameter_metrics(pool,pool,'x',{'type':'categorical'},2)
        self.assertEqual(row['retained_categories'],5)
        self.assertAlmostEqual(h.sum(),1.)
        pool['param__x']=[179.,0.,1.,178.,2.]
        row,_,_=parameter_metrics(pool,pool,'x',{},2,period=180.)
        self.assertLess(row['width'],5.)
        self.assertGreater(row['circular_resultant'],.99)
        self.assertTrue(np.isnan(row['fraction_near_boundary']))

if __name__=='__main__':unittest.main()
