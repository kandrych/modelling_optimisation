"""Checks for the new read-only fit notebook's scientific comparison rules."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/post-processing'))
import fit_assessment_utils as fa
from model_analysis_utils import GROUPS


class FitAssessmentTests(unittest.TestCase):
    def pool(self):
        data = pd.DataFrame(dict(evaluation_id=['a', 'b', 'c', 'd'],
            parameter_hash=['x', 'x', 'y', 'z'], history=['h']*4,
            physical_pool_key=['physical']*4, score_pool_key=['score']*4,
            param__x=[.1, .1, .8, .9], param__category=['a', 'a', 'b', 'b']))
        for columns in GROUPS.values():
            for column in columns:
                data[column] = 0.
        data['score_sed'] = [1., 9., 3., 5.]
        data['score_alma'] = [8., 8., 1., 2.]
        return data

    def references(self):
        return {'x': dict(kind='linear', lower=0., upper=1.),
                'category': dict(kind='categorical', categories=['a', 'b'])}

    def test_unique_count_aggregates_before_selection_and_preserves_input(self):
        pool = self.pool()
        original = pool.copy(deep=True)
        selected, cutoff = fa.analysis_sample(pool, pool.score_sed, .34, 'configurations')
        self.assertEqual(set(selected.parameter_hash), {'x', 'y', 'z'})  # Ties at 5 retained.
        self.assertEqual(cutoff, 5.)
        self.assertEqual(selected.set_index('parameter_hash').loc['x', 'analysis_score'], 5.)
        self.assertEqual(len(selected), 3)
        pd.testing.assert_frame_equal(pool, original)

    def test_weights_half_delta_zero_minimum_and_weighted_quantile(self):
        pool = self.pool().iloc[:2].copy()
        pool['analysis_score'] = [10., 12.5]
        weights = fa.sample_weights(pool, .25)
        self.assertAlmostEqual(weights[1] / weights[0], .5)
        pool['analysis_score'] = [0., 1.]
        with self.assertRaises(ValueError):
            fa.sample_weights(pool, .25)
        self.assertAlmostEqual(fa.sample_weights(pool, .25, 1.)[1], 1/3)
        pool.loc[pool.index[1], 'param__x'] = .9
        row = fa.weighted_summary(pool, 'x', self.references()['x'], [.95, .05])
        self.assertEqual(row['q90'], .1)

    def test_missing_parameter_excluded_with_unique_support_count(self):
        pool = self.pool()
        pool.loc[2, 'param__x'] = np.nan
        row = fa.weighted_summary(pool, 'x', self.references()['x'], [.25]*4)
        self.assertEqual(row['n_valid'], 3)
        self.assertEqual(row['n_unique'], 2)

    def test_minimum_ties_deterministic_and_pool_choice_explicit(self):
        pool = self.pool()
        pool['score_sed'] = [1., 1., 2., 3.]
        model, ties, score = fa.winner(pool.iloc[::-1], ['SED'])
        self.assertEqual((model, ties, score), ('a', 2, 1.))
        self.assertEqual(len(fa.choose_pool({'b': pool, 'a': pool.iloc[:2]})), 4)
        self.assertEqual(len(fa.choose_pool({'b': pool, 'a': pool.iloc[:2]}, 'a')), 2)
        self.assertTrue(fa.choose_pool({}).empty)

    def test_additions_use_union_finite_pool_on_both_sides(self):
        pool = self.pool()
        pool.loc[0, 'score_alma'] = np.nan  # Best SED model cannot enter SED→SED+ALMA.
        seen_groups = []
        def partitions(candidates, records, groups):
            seen_groups.append(tuple(groups))
            columns = [col for group in groups for col in GROUPS[group]]
            complete = candidates.loc[np.isfinite(candidates[columns]).all(axis=1)]
            return {'k': complete}
        with patch.object(fa, 'available_score_pools', side_effect=partitions):
            effects, pairs, _ = fa.addition_evidence(pool, {}, {}, self.references(), (.5,), 1)
        pair = pairs.loc[pairs.before.eq('SED') & pairs.after.eq('SED+ALMA')].iloc[0]
        self.assertNotEqual(pair.before_id, 'a')
        self.assertEqual(pair.n_pool, 3)
        self.assertIn(('SED', 'ALMA'), seen_groups)
        self.assertTrue(effects.supported.all())

    def test_robustness_reports_zero_minimum_without_inventing_scale(self):
        pool = self.pool()
        pool['score_sed'] = 0.
        table = fa.robustness_table(pool, ['SED'], self.references(), (.4,), (None, .25))
        self.assertEqual(table.loc[table.half_fraction.eq('equal'), 'status'].unique().tolist(), ['ok'])
        self.assertTrue(table.loc[table.half_fraction.eq('0.25'), 'status'].str.contains('Zero').all())

    def test_realization_mismatch_and_ambiguous_legacy_folders(self):
        row = pd.DataFrame([dict(evaluation_id='a', seed=2, instance='null', cost=10.)])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder/'trial_status.json').write_text(json.dumps(dict(seed=1, instance=None, cost=10.)))
            with patch.object(fa, 'resolve_model_files', return_value=[{'folder': folder}]):
                folders, audit = fa.resolve_diagnostics(row, {})
            self.assertFalse(folders)
            self.assertEqual(audit.iloc[0].rejected_matches, 1)
            (folder/'trial_status.json').unlink()
            with patch.object(fa, 'resolve_model_files', return_value=[{'folder': folder}]*2):
                folders, audit = fa.resolve_diagnostics(row, {})
            self.assertFalse(folders)
            self.assertEqual(audit.iloc[0].status, 'ambiguous folders')

    def test_sed_units_residuals_and_score_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'data_th').mkdir()
            fits.HDUList([fits.PrimaryHDU(np.array([1., 2., 3., 4.]).reshape(1,1,1,4)/1e3),
                          fits.ImageHDU(np.array([1.,2.,3.,4.]))]).writeto(root/'data_th/sed_rt.fits.gz')
            phot = root/'observed.phot'
            lines = ['header']*4
            for wave, value in [(20000.,3.), (30000.,5.), (40000.,7.)]:
                lines.append(' '.join(map(str, [0,0,0,'unit','band','source',0,0,0,
                    wave,value/wave,1/wave,'unit',0,True,0,'bib','comment'])))
            phot.write_text('\n'.join(lines))
            law = root/'law.dat'
            law.write_text('header\nheader\nheader\n10000 0\n40000 0\n')
            points, audit = fa.sed_residuals(root, {'ebminv': 0., 'chi2': 14.}, phot, law, secondary=False)
            np.testing.assert_allclose(points.residual_sigma, [1.,2.,3.])
            np.testing.assert_allclose(points.wavelength_um, [2.,3.,4.])
            self.assertEqual(audit['status'], 'score verified')
            points, audit = fa.sed_residuals(root, {'ebminv': 0., 'chi2': 20.}, phot, law, secondary=False)
            self.assertTrue(points.empty)
            self.assertEqual(audit['status'], 'SED recipe/data mismatch')


if __name__ == '__main__':
    unittest.main()
