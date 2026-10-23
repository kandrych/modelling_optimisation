"""Check scientific weighting invariants and the actual ChainConsumer interface."""
import importlib.util
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/post-processing'))
from score_weighted_pool import score_weighted_pool, plot_score_weighted_pool, group_parameter_references


class ScoreWeightTests(unittest.TestCase):
    def test_physical_groups_retain_types_unknowns_and_reject_duplicates(self):
        refs = {name: {'kind': 'linear'} for name in (
            'inclination', 'disk_Rmid', 'zone_2_-gamma_exp', 'disk_total_dust_mass',
            'zone_1_species_1_porosity', 'new_parameter')}
        refs['alpha_viscosity'] = {'kind': 'categorical', 'categories': ['0.01', '0.1']}
        groups = group_parameter_references(refs)
        self.assertEqual(sum(map(len, groups.values())), len(refs))
        self.assertIn('disk_Rmid', groups['Disc geometry and orientation'])
        self.assertIn('zone_2_-gamma_exp', groups['Disc geometry and orientation'])
        self.assertIs(groups['Dust distribution, settling and heating']['alpha_viscosity'], refs['alpha_viscosity'])
        self.assertIn('new_parameter', groups['Other parameters'])
        self.assertNotIn('zone_2_Rin', {name for group in groups.values() for name in group})
        with self.assertRaisesRegex(ValueError, 'multiple groups'):
            group_parameter_references(refs, {'first': ['disk_Rmid'], 'second': ['disk_Rmid']})
        custom = group_parameter_references(refs, {'Chosen': ['disk_Rmid']})
        self.assertEqual(len(custom['Other parameters']), len(refs)-1)

    def pool(self, scores=(20., 22., 24.)):
        n = len(scores)
        return pd.DataFrame(dict(evaluation_id=[f'e{i}' for i in range(n)],
            parameter_hash=[f'p{i}' for i in range(n)], history=['h']*n,
            score_pool_key=['compatible']*n, physical_pool_key=['physical']*n,
            score_alma=scores))

    def test_half_weight_ratio_normalisation_and_original_unchanged(self):
        pool = self.pool()
        original = pool.copy(deep=True)
        weights, diagnostics = score_weighted_pool(pool, 'ALMA', ['ALMA'], (.1,))
        np.testing.assert_allclose(weights.weight, np.array([1., .5, .25])/1.75)
        self.assertAlmostEqual(diagnostics.temperature.iloc[0], 2./np.log(2.))
        self.assertAlmostEqual(weights.weight.sum(), 1.)
        pd.testing.assert_frame_equal(pool, original)

    def test_absolute_scale_offset_and_score_unit_invariance(self):
        pool = self.pool()
        original, _ = score_weighted_pool(pool, 'ALMA', ['ALMA'], absolute_half_weight_deltas=[2.])
        shifted, _ = score_weighted_pool(pool.assign(score_alma=pool.score_alma-100),
            'ALMA', ['ALMA'], absolute_half_weight_deltas=[2.])
        scaled, _ = score_weighted_pool(pool.assign(score_alma=pool.score_alma*100),
            'ALMA', ['ALMA'], absolute_half_weight_deltas=[200.])
        np.testing.assert_allclose(original.weight, shifted.weight)
        np.testing.assert_allclose(original.weight, scaled.weight)

    def test_zero_best_requires_absolute_and_underflow_is_reported(self):
        pool = self.pool([0., 1., 1e300])
        with self.assertRaisesRegex(ValueError, 'Best score is zero'):
            score_weighted_pool(pool, 'ALMA', ['ALMA'])
        weights, diag = score_weighted_pool(pool, 'ALMA', ['ALMA'], absolute_half_weight_deltas=[1.])
        np.testing.assert_allclose(weights.weight, [2/3, 1/3, 0])
        self.assertEqual(diag.n_zero_weights.iloc[0], 1)
        self.assertEqual(len(weights), 3)

    def test_repeats_remain_evaluations_and_concentration_is_aggregated(self):
        pool = self.pool([1., 1., 1.]).assign(parameter_hash=['same', 'same', 'other'])
        weights, diag = score_weighted_pool(pool, 'ALMA', ['ALMA'], (.1,))
        np.testing.assert_allclose(weights.weight, [1/3]*3)
        self.assertAlmostEqual(diag.evaluation_weight_ess.iloc[0], 3)
        self.assertAlmostEqual(diag.configuration_weight_ess.iloc[0], 1.8)

    def test_combined_objective_keeps_original_relative_weights(self):
        pool = self.pool([1., 3., 5.]).assign(score_sed=[10., 9., 8.])
        weights, _ = score_weighted_pool(pool, 'SED+ALMA', ['SED', 'ALMA'],
                                         absolute_half_weight_deltas=[1.])
        np.testing.assert_allclose(weights.objective_score, [11., 12., 13.])
        np.testing.assert_allclose(weights.weight, np.array([1., .5, .25])/1.75)

    def test_reject_mixed_pools_duplicate_ids_and_invalid_temperatures(self):
        pool = self.pool()
        for field, values in [('score_pool_key', ['a', 'b', 'b']),
                              ('physical_pool_key', ['a', 'b', 'b']),
                              ('evaluation_id', ['a', 'a', 'b'])]:
            with self.assertRaises(ValueError):
                score_weighted_pool(pool.assign(**{field: values}), 'ALMA', ['ALMA'])
        for values in [[], [0.], [-1.], [np.inf], [np.nan], [1., 1.]]:
            with self.assertRaises(ValueError):
                score_weighted_pool(pool, 'ALMA', ['ALMA'], absolute_half_weight_deltas=values)

    @unittest.skipUnless(importlib.util.find_spec('chainconsumer'), 'Optional plotting dependency missing')
    def test_grouped_plots_share_missing_coordinate_support_and_reference(self):
        rng = np.random.default_rng(5)
        pool = self.pool(np.arange(200)+20.)
        pool['param__x'], pool['param__y'] = rng.normal(size=(2, 200))
        pool['param__category'] = ['a', 'b']*100
        pool.loc[0, 'param__x'] = np.nan
        refs = {'x': {'kind': 'linear'}, 'y': {'kind': 'linear'},
                'category': {'kind': 'categorical', 'categories': ['a', 'b']}}
        weights, diag = score_weighted_pool(pool, 'ALMA', ['ALMA'], absolute_half_weight_deltas=[200.])
        reports = []
        for label, names in {'First': ['x'], 'Second': ['y', 'category']}.items():
            figures, report = plot_score_weighted_pool(pool, weights, diag,
                {name: refs[name] for name in names}, style='reference', smooth=2,
                parameter_group=label, complete_references=refs)
            reports.append(report)
            self.assertTrue(report.parameter_group.eq(label).all())
            for fig in figures.values():
                self.assertIn(label, fig._suptitle.get_text())
                fig.canvas.draw()
                plt.close(fig)
        numeric = pd.concat(reports).query("parameter == 'numeric_corner'")
        self.assertTrue(numeric.n_valid.eq(199).all())
        self.assertTrue(numeric.reference_evaluation_id.eq('e1').all())
        self.assertEqual(numeric.retained_weight.nunique(), 1)
        self.assertEqual(numeric.score_color_min.nunique(), 1)
        bars = reports[1].query("status == 'category_bar'")
        self.assertTrue(bars.n_valid.eq(199).all())
        self.assertAlmostEqual(bars.weight.sum(), 1.)

    @unittest.skipUnless(importlib.util.find_spec('chainconsumer'), 'Optional plotting dependency missing')
    def test_reference_style_scores_marker_and_descriptive_summaries(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=400)
        y = .7*x+rng.normal(scale=.5, size=400)
        pool = self.pool(20+x*x+y*y)
        pool['param__x'], pool['param__y'] = x, y
        # The global best lacks a displayed coordinate: mark the best DISPLAYED model.
        pool.loc[0, 'score_alma'] = 0.1
        pool.loc[0, 'param__x'] = np.nan
        pool.loc[1, 'score_alma'] = 1.
        original = pool.copy(deep=True)
        weights, diag = score_weighted_pool(pool, 'ALMA', ['ALMA'],
            absolute_half_weight_deltas=[20., 40., 80.])
        refs = {'x': {'kind': 'linear'}, 'y': {'kind': 'linear'}}
        figures, coverage = plot_score_weighted_pool(pool, weights, diag, refs,
            contour_masses=(.68, .95, .997), smooth=2, style='reference')
        numeric = coverage.loc[coverage.parameter.eq('numeric_corner')].iloc[0]
        self.assertEqual(numeric.setting, 'half_weight_02')
        self.assertEqual(numeric.reference_evaluation_id, 'e1')
        self.assertEqual(numeric.reference_score, 1.)
        summaries = coverage.loc[coverage.status.eq('descriptive_marginal_summary')]
        self.assertEqual(set(summaries.parameter), {'x', 'y'})
        self.assertTrue((summaries.lower <= summaries.center).all())
        self.assertTrue((summaries.center <= summaries.upper).all())
        self.assertTrue(summaries.summary_statistic.eq('hdi').all())
        fig = figures['corner']
        fig.canvas.draw()
        self.assertEqual(len(fig.axes), 5)  # 2x2 layout (one unused axis) plus score colourbar.
        self.assertTrue(any('ALMA score (log colour scale; lower is better)' in ax.get_ylabel() for ax in fig.axes))
        star_positions = [collection.get_offsets()[0] for ax in fig.axes for collection in ax.collections
                          if hasattr(collection, 'get_sizes') and np.array_equal(collection.get_sizes(), [100])]
        self.assertEqual(len(star_positions), 1)
        np.testing.assert_allclose(star_positions[0], [x[1], y[1]])
        plt.close(fig)
        pd.testing.assert_frame_equal(pool, original)
        with self.assertRaisesRegex(ValueError, 'existing weighting setting'):
            plot_score_weighted_pool(pool, weights, diag, refs, style='reference', reference_setting='unknown')

    @unittest.skipUnless(importlib.util.find_spec('chainconsumer'), 'Optional plotting dependency missing')
    def test_multimodal_marginal_exports_separate_hdi_bands(self):
        rng = np.random.default_rng(42)
        pool = self.pool(np.ones(1000))
        pool['param__x'] = np.r_[rng.normal(-4., .3, 500), rng.normal(4., .3, 500)]
        weights, diagnostics = score_weighted_pool(pool, 'ALMA', ['ALMA'])
        figures, coverage = plot_score_weighted_pool(pool, weights, diagnostics,
            {'x': {'kind': 'linear'}}, style='reference', smooth=2)
        bands = coverage.loc[coverage.status.eq('descriptive_marginal_summary')].sort_values('lower')
        self.assertEqual(len(bands), 2)
        self.assertLess(bands.iloc[0].upper, bands.iloc[1].lower)
        self.assertTrue(bands.summary_mass_scope.eq('all_bands_together').all())
        self.assertTrue(bands.center_kind.eq('band_mode').all())
        for fig in figures.values():
            fig.canvas.draw()
            plt.close(fig)

    @unittest.skipUnless(importlib.util.find_spec('chainconsumer'), 'Optional plotting dependency missing')
    def test_render_missing_coordinates_categories_and_low_support(self):
        rng = np.random.default_rng(4)
        n = 200
        pool = self.pool(np.ones(n))
        pool['param__x'] = rng.normal(size=n)
        pool['param__y'] = 10**rng.normal(size=n)
        pool['param__category'] = ['a', 'b']*(n//2)
        pool['param__constant'] = 3.
        pool.loc[0, 'param__x'] = np.nan
        pool.loc[1, 'param__y'] = 0.  # Invalid log coordinate.
        refs = {'x': {'kind': 'linear'}, 'y': {'kind': 'log10'},
                'category': {'kind': 'categorical', 'categories': ['a', 'b', 'unvisited']},
                'constant': {'kind': 'linear'}}
        weights, diag = score_weighted_pool(pool, 'ALMA', ['ALMA'])
        figures, coverage = plot_score_weighted_pool(pool, weights, diag, refs, bins=10)
        self.assertEqual(set(figures), {'corner', 'categories'})
        numeric = coverage.loc[coverage.parameter.eq('numeric_corner')]
        np.testing.assert_allclose(numeric.retained_weight, .99)
        self.assertTrue(numeric.n_valid.eq(198).all())
        self.assertTrue(coverage.loc[coverage.parameter.eq('constant')].status.eq('constant_or_unavailable').all())
        bars = coverage.loc[coverage.status.eq('category_bar')]
        np.testing.assert_allclose(bars.groupby('setting').weight.sum(), 1.)
        self.assertTrue(bars.loc[bars.category.eq('unvisited')].weight.eq(0.).all())
        for fig in figures.values():
            fig.canvas.draw()
            plt.close(fig)
        pool['score_alma'] = [1.] + [1e6]*(n-1)
        weights, diag = score_weighted_pool(pool, 'ALMA', ['ALMA'], (.1,))
        figures, coverage = plot_score_weighted_pool(pool, weights, diag, {'y': refs['y']})
        self.assertFalse(figures)
        self.assertTrue(coverage.status.str.startswith('insufficient').all())


if __name__ == '__main__':
    unittest.main()
