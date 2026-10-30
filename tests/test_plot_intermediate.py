"""Plot switches must leave scientific scores and convolution results unchanged."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from astropy.io import fits

import SMAC3
from lib import obriy_mcfost as mcfost
from lib import obriy_polarimetry as pdi


class PlotIntermediateTests(unittest.TestCase):
    def test_sed_trial_without_plots_and_final_with_plots_have_same_score(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(mcfost, 'run_mcfost'), \
                patch.object(mcfost, 'plot_mcfost_disk_structure'), \
                patch.object(mcfost, 'plot_mcfost_density_temperature_cuts'):
            work = Path(directory)
            (work / 'data_th').mkdir()
            wave = np.geomspace(.2, 1000., 50)
            flux = 1e-9 / (1 + wave)
            fits.HDUList([
                fits.PrimaryHDU(np.stack([flux, .5 * flux])[:, None, None, :] / 1e3),
                fits.ImageHDU(wave),
            ]).writeto(work / 'data_th/sed_rt.fits.gz')
            data = {'sed': (wave, flux, .1 * flux)}
            fidelity = {'stage': 'sed', 'products': ['sed']}
            args = SimpleNamespace(plot_intermediate=False, puffed_up_rim=False)
            trial = mcfost.load_and_score_outputs(fidelity, work, data, args, {})
            self.assertEqual(list(work.rglob('*.png')), [])
            final = SMAC3.final_model(
                {}, 1, 1., fidelity_result=fidelity, par_path=work / 'model.para',
                results_dir=work, data_arg=data, args=args, incumbent={})
            self.assertEqual(trial, final)
            self.assertTrue((work / 'SED_Akke_model1_MCFOST.png').is_file())
            self.assertTrue((work / 'SED_secondary_comparison.png').is_file())

    def pdi_images(self, size=32):
        yy, xx = np.indices((size, size))
        centre = (size - 1) / 2
        phi = np.arctan2(yy - centre, xx - centre)
        intensity = 2 + np.exp(-((xx - centre)**2 + (yy - centre)**2) / 50)
        q, u = -.1 * intensity * np.cos(2 * phi), -.1 * intensity * np.sin(2 * phi)
        qp, up, pi, _ = pdi.compute_qphi_uphi_pi(q, u)
        return dict(I=intensity, Q=q, U=u, Q_phi=qp, U_phi=up, PI=pi)

    def test_pdi_comparison_plots_follow_flag_for_all_bands(self):
        images = self.pdi_images()
        model = dict(img_tot=images['I'], img_q=images['Q'], img_u=images['U'],
                     q_phi=images['Q_phi'], pixel_scale_mas=3.6)
        corrected_model = dict(model, q_phi=.8 * model['q_phi'],
                               img_q=.8 * model['img_q'], img_u=.8 * model['img_u'])
        results = {'mcfost_convolved': model,
                   'mcfost_convolved_unresolved_corrected': corrected_model}
        observations = self.pdi_images(size=48)
        grid = dict(grid_nx=32, grid_ny=32, grid_size=115.2, distance_pc=1000.)
        prepared = pdi.prepare_pdi_observation(observations, 3.6, grid)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(pdi, 'polarimetric_analysis', return_value=results) as analysis, \
                patch.object(pdi, 'plot_pdi_constraints') as comparison_plot, \
                patch.object(pdi, 'plot_differential_quadrants_slides') as quadrant_plot, \
                patch.object(pdi, 'plot_pdi_azimuthal_comparison', wraps=pdi.plot_pdi_azimuthal_comparison) as azimuthal_plot, \
                patch.object(pdi, 'profile_chi2') as legacy_profiles, \
                patch.object(pdi, 'plot_polarimetric_image'), \
                patch.object(pdi, 'plot_image_grid', return_value=(Mock(), [])), \
                patch.object(mcfost, 'plot_mcfost_disk_structure'), \
                patch.object(mcfost, 'plot_mcfost_density_temperature_cuts'):
            work = Path(directory)
            (work / 'data_th').mkdir()
            (work / 'data_th/sed_rt.fits.gz').touch()
            for band, corrected in ((band, corrected) for band in ('I', 'V', 'H')
                                    for corrected in (False, True)):
                scores = []
                for enabled in (False, True):
                    with self.subTest(band=band, corrected=corrected, plot=enabled):
                        comparison_plot.reset_mock()
                        quadrant_plot.reset_mock()
                        azimuthal_plot.reset_mock()
                        score, info = mcfost.load_and_score_outputs(
                            {'stage': band, 'products': [f'pdi_{band}']}, work,
                            {f'pdi_{band}': {'pol_images': observations, 'psf': images['I'],
                                            'prepared_observation': prepared}},
                            SimpleNamespace(plot_intermediate=enabled,
                                            correct_unresolved_polarimetry=corrected,
                                            unresolved_correction_radius_px=3.), {})
                        self.assertEqual(comparison_plot.call_count, int(enabled))
                        self.assertEqual(quadrant_plot.call_count, int(enabled))
                        self.assertEqual(azimuthal_plot.call_count, int(enabled))
                        self.assertEqual(analysis.call_args.kwargs['correct_unresolved'], corrected)
                        legacy_profiles.assert_not_called()
                        self.assertNotIn('diagnostic_plot_errors', info['pdi'][band]['constraints'])
                        if enabled:
                            call = azimuthal_plot.call_args.args
                            self.assertIs(call[0], observations['Q_phi'])
                            self.assertIs(call[1], (corrected_model if corrected else model)['q_phi'])
                            self.assertEqual(call[3], info['pdi'][band]['constraints']['aperture_radius_mas'])
                            self.assertIs(azimuthal_plot.call_args.kwargs['observed_profile'], prepared['azimuthal_profile'])
                            self.assertTrue((work / 'figures' / f'pdi_constraints_{band.lower()}_azimuthal.png').is_file())
                        scores.append((score, info['pdi'][band]['constraints']))
                self.assertEqual(scores[0], scores[1])

    def test_azimuthal_profiles_use_shared_circle_and_preserve_signed_flux(self):
        model = np.ones((32, 32))
        model[16:, 16:] = -.1
        observed = np.pad(model, 8, constant_values=np.nan)
        # The larger image contains extra emission outside the shared circle.
        observed[:8, :] = 1e6
        profiles = [pdi.measure_pdi_azimuthal_profile(image, 3.6, 57.6)
                    for image in (observed, model)]
        np.testing.assert_array_equal(profiles[0]['theta_deg_centers'], profiles[1]['theta_deg_centers'])
        np.testing.assert_allclose(profiles[0]['value'], profiles[1]['value'])
        self.assertAlmostEqual(profiles[0]['value'].sum(), 1.)
        self.assertTrue(np.any(profiles[0]['value'] < 0))
        self.assertEqual(profiles[0]['aperture_radius_mas'], 57.6)

    def test_pdi_plot_failures_are_reported_without_changing_scores(self):
        images = self.pdi_images()
        model = dict(img_tot=images['I'], img_q=images['Q'], img_u=images['U'],
                     q_phi=images['Q_phi'])
        baseline = pdi.compare_pdi_constraints(images, model, 3.6)
        for failing_function, filename in (
                ('plot_pdi_constraints', 'pdi_constraints_v.png'),
                ('plot_pdi_azimuthal_comparison', 'pdi_constraints_v_azimuthal.png'),
                ('plot_differential_quadrants_slides', 'pdi_constraints_v_model_quadrants_slides.png')):
            with self.subTest(function=failing_function), tempfile.TemporaryDirectory() as directory, \
                    redirect_stdout(io.StringIO()), \
                    patch.object(pdi, 'plot_pdi_constraints'), \
                    patch.object(pdi, 'plot_pdi_azimuthal_comparison'), \
                    patch.object(pdi, 'plot_differential_quadrants_slides'):
                def fail(*args, **kwargs):
                    pdi.plt.figure()
                    raise OSError('Cannot save diagnostic')
                getattr(pdi, failing_function).side_effect = fail
                figures_before = pdi.plt.get_fignums()
                result = pdi.compare_pdi_constraints(
                    images, model, 3.6, output_path=Path(directory) / 'pdi_constraints_v.png', band='V')
                errors = result.pop('diagnostic_plot_errors')
                self.assertEqual(len(errors), 1)
                self.assertIn(filename, errors[0])
                self.assertIn('OSError: Cannot save diagnostic', errors[0])
                self.assertEqual(result, baseline)
                self.assertEqual(pdi.plt.get_fignums(), figures_before)
                self.assertEqual(pdi.plot_pdi_constraints.call_count, 1)
                self.assertEqual(pdi.plot_pdi_azimuthal_comparison.call_count, 1)
                self.assertEqual(pdi.plot_differential_quadrants_slides.call_count, 1)

    def test_pdi_convolution_forwards_plot_without_changing_images(self):
        images = self.pdi_images()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            work = Path(directory)
            (work / 'data_0.82').mkdir()
            header = fits.Header({'CDELT2': 3.6 / 3.6e6})
            fits.PrimaryHDU(images['I'], header).writeto(work / 'data_0.82/RT.fits.gz')
            loaded = (None, header, images['I'], images['Q'], images['U'],
                      images['I'], images['I'], images['I'], images['I'], images['I'])
            for mode in ('synthetic', 'file'):
                outputs = []
                for corrected, enabled in ((c, p) for c in (False, True) for p in (False, True)):
                    with self.subTest(mode=mode, corrected=corrected, plot=enabled), \
                            patch.object(pdi, 'load_mcfost_images_1wave', return_value=loaded), \
                            patch.object(pdi, 'convolve_polarimetric_images', wraps=pdi.convolve_polarimetric_images) as convolution, \
                            patch.object(pdi, 'calculate_unresolved', wraps=pdi.calculate_unresolved) as correction, \
                            patch.object(pdi, 'plot_polarimetric_image') as image_plot, \
                            patch.object(pdi, 'plot_image_grid', side_effect=lambda **k: pdi.plt.subplots()), \
                            patch.object(pdi, 'deconvolution', side_effect=lambda image, *a, **k: image) as deconvolution:
                        result = pdi.polarimetric_analysis(
                            str(work), .82, camera='zimpol', convolution_mode=mode,
                            psf_array=np.ones((3, 3)), psf_cut=None, plot=enabled,
                            correct_unresolved=corrected,
                            extra_title='', fig_dir=str(work) + '/')
                        self.assertEqual(convolution.call_args.kwargs['plot'], enabled)
                        self.assertEqual(image_plot.called, enabled)
                        self.assertEqual(correction.call_count, int(corrected or enabled) + int(enabled))
                        self.assertEqual(deconvolution.call_count, 3 if enabled else 0)
                        self.assertEqual('mcfost_not_convolved_unresolved_corrected' in result, enabled)
                        self.assertEqual('mcfost_convolved_unresolved_corrected' in result, corrected or enabled)
                        self.assertNotIn('diagnostic_plot_errors', result)
                        if corrected or enabled:
                            raw = result['mcfost_convolved']
                            radius, *_ = pdi.compute_grid(raw['img_q'])
                            _, _, expected_q, expected_u = correction._mock_wraps(
                                3., raw['img_q'], raw['img_u'], raw['img_tot'], 3.6, radius, 100)
                            actual = result['mcfost_convolved_unresolved_corrected']
                            np.testing.assert_array_equal(actual['img_q'], expected_q)
                            np.testing.assert_array_equal(actual['img_u'], expected_u)
                        outputs.append(result['mcfost_convolved'])
                for output in outputs[1:]:
                    for key in ('img_tot', 'img_q', 'img_u', 'q_phi'):
                        np.testing.assert_array_equal(outputs[0][key], output[key])

    def test_unused_correction_failures_do_not_reject_uncorrected_results(self):
        images = self.pdi_images()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            work = Path(directory)
            (work / 'data_0.82').mkdir()
            header = fits.Header({'CDELT2': 3.6 / 3.6e6})
            fits.PrimaryHDU(images['I'], header).writeto(work / 'data_0.82/RT.fits.gz')
            loaded = (None, header, images['I'], images['Q'], images['U'],
                      images['I'], images['I'], images['I'], images['I'], images['I'])
            with patch.object(pdi, 'load_mcfost_images_1wave', return_value=loaded), \
                    patch.object(pdi, 'plot_polarimetric_image'), \
                    patch.object(pdi, 'deconvolution', side_effect=lambda image, *a, **k: image), \
                    patch.object(pdi, 'calculate_unresolved', side_effect=ValueError('Invalid correction aperture')) as correction:
                def analyse(corrected, plot):
                    return pdi.polarimetric_analysis(
                        str(work), .82, camera='zimpol', convolution_mode='file',
                        psf_array=np.ones((3, 3)), psf_cut=None, plot=plot,
                        correct_unresolved=corrected, extra_title='', fig_dir=str(work / 'figures') + '/')

                without_plots = analyse(False, False)
                correction.assert_not_called()
                self.assertFalse((work / 'figures').exists())
                with_plots = analyse(False, True)
                self.assertEqual(len(with_plots['diagnostic_plot_errors']), 2)
                self.assertNotIn('mcfost_convolved_unresolved_corrected', with_plots)
                for key in ('img_tot', 'img_q', 'img_u', 'q_phi'):
                    np.testing.assert_array_equal(without_plots['mcfost_convolved'][key],
                                                  with_plots['mcfost_convolved'][key])
                for plot in (False, True):
                    with self.assertRaisesRegex(ValueError, 'Invalid correction aperture'):
                        analyse(True, plot)


if __name__ == '__main__':
    unittest.main()
