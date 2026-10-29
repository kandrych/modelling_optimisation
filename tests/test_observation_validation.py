"""Startup validation without workers, MCFOST or real observation files."""
import __future__
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from astropy.io import fits

from lib import obriy_alma as alma
from lib import obriy_polarimetry as pdi
from lib import obriy_mcfost as mcfost
from lib.obriy_fidelity import fidelity_products
from lib.obriy_interferometry_data import load_interferometry_data
from lib.obriy_observations import (
    load_sed_observation, load_pdi_image, validate_pdi_observation,
    load_alma_observation, pdi_scoring_aperture,
)


class ObservationValidationTests(unittest.TestCase):
    def test_sed_invalid_arrays(self):
        invalid = [
            ([], [], []), ([1], [2], [.1]),
            ([1, 2], [3], [.1, .1]), ([[1, 2]], [[3, 4]], [[.1, .1]]),
            ([0, 2], [3, 4], [.1, .1]), ([-1, 2], [3, 4], [.1, .1]),
            ([1, 2], [3, 4], [0, .1]), ([1, 2], [3, 4], [-1, .1]),
            ([1, np.nan], [3, 4], [.1, .1]),
            ([1, 2], [3, np.inf], [.1, .1]),
            ([1, 2], [3, 4], [.1, np.nan]),
        ]
        for columns in invalid:
            with self.subTest(columns=columns), self.assertRaisesRegex(ValueError, 'SED.*example.phot') as caught:
                load_sed_observation('/data/example.phot', lambda path: columns)
            self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_valid_sed_and_reader_error(self):
        columns = ([1, 2], [-3, 4], [.1, .2])
        result = load_sed_observation('/data/example.phot', lambda path: columns)
        np.testing.assert_array_equal(result, columns)
        original = FileNotFoundError('file missing')
        with self.assertRaisesRegex(ValueError, 'SED.*example.phot') as caught:
            load_sed_observation('/data/example.phot', Mock(side_effect=original))
        self.assertIs(caught.exception.__cause__, original)

    def pdi_fixture(self):
        # A positive Qphi field and its consistent Cartesian Stokes images.
        yy, xx = np.indices((220, 220))
        phi = np.arctan2(yy - 109.5, xx - 109.5)
        images = {key: np.ones((220, 220)) for key in ('I', 'PI', 'Q_phi', 'U_phi')}
        images['U_phi'] *= 0
        images['Q'] = -np.cos(2 * phi)
        images['U'] = -np.sin(2 * phi)
        return images, np.ones((220, 220))

    def validate_pdi(self, images, psf, scale=3.6):
        validate_pdi_observation(images, psf, scale, 'V', {'Q_phi': '/data/qphi.fits'})

    def test_pdi_missing_file_and_image_payloads(self):
        original = OSError('missing FITS')
        with self.assertRaisesRegex(ValueError, 'PDI V Q_phi.*qphi.fits') as caught:
            load_pdi_image('/data', 'qphi.fits', Mock(side_effect=original), 'PDI V Q_phi')
        self.assertIs(caught.exception.__cause__, original)
        for fault in ('missing', 'none', 'empty', 'psf'):
            images, psf = self.pdi_fixture()
            if fault == 'missing':
                del images['Q_phi']
            elif fault == 'none':
                images['Q_phi'] = None
            elif fault == 'empty':
                images['Q_phi'] = np.empty((0, 0))
            else:
                psf = None
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, 'PDI V.*qphi.fits'):
                self.validate_pdi(images, psf)

    def test_pdi_accepts_partial_finite_apertures_and_zero_intensity_sum(self):
        for value in (0., 1.):
            with self.subTest(value=value):
                images, psf = self.pdi_fixture()
                for array in images.values():
                    array[:] = np.nan
                    array[110, 110] = value
                original = {key: array.copy() for key, array in images.items()}
                self.validate_pdi(images, psf)
                for key, array in images.items():
                    np.testing.assert_array_equal(array, original[key])
                result = load_pdi_image('/data', 'qphi.fits',
                                        Mock(return_value=images['Q_phi']), 'PDI V Q_phi')
                self.assertIs(result, images['Q_phi'])

    def test_pdi_rejects_missing_finite_aperture_samples(self):
        for key in ('I', 'PI', 'Q_phi', 'U_phi', 'Q', 'U'):
            images, psf = self.pdi_fixture()
            _, _, aperture = pdi_scoring_aperture(images[key].shape, 3.6)
            images[key][aperture] = np.nan
            # Finite values outside the aperture are insufficient.
            with self.subTest(component=key), self.assertRaisesRegex(ValueError, 'no finite samples'):
                self.validate_pdi(images, psf)

    def test_pdi_rejects_negative_or_nonfinite_intensity_sum(self):
        for value in (-1., np.inf, -np.inf, np.finfo(float).max):
            images, psf = self.pdi_fixture()
            images['I'][:] = value
            images['I'][110, 110] = 0  # There is at least one finite sample.
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'sum_I.*finite and nonnegative'):
                self.validate_pdi(images, psf)

    def test_pdi_requires_compatible_grid_and_valid_pixel_scale(self):
        images, psf = self.pdi_fixture()
        for scale in (0., -1., np.nan):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, 'pixel scale'):
                self.validate_pdi(images, psf, scale)
        images['U'] = images['U'][:-1]
        with self.assertRaisesRegex(ValueError, 'same 2D pixel grid'):
            self.validate_pdi(images, psf)

    def test_pdi_ignores_pixels_outside_scoring_aperture(self):
        images, psf = self.pdi_fixture()
        expected = pdi.measure_pdi_constraints(images, 3.6)
        _, _, aperture = pdi_scoring_aperture(images['I'].shape, 3.6)
        for array in images.values():
            array[~aperture] = np.nan
        self.validate_pdi(images, psf)
        actual = pdi.measure_pdi_constraints(images, 3.6)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], err_msg=key)
        images['Q_phi'][110, 110] = np.nan
        with self.assertRaisesRegex(ValueError, 'scoring aperture'):
            pdi.measure_pdi_constraints(images, 3.6)

    def alma_fixture(self):
        yy, xx = np.indices((64, 64))
        image = np.where((xx + yy) % 2, .01, -.01).astype(np.float32)
        image[np.hypot(xx - 31.5, yy - 31.5) < 4] = 1
        header = fits.Header(dict(
            NAXIS=4, NAXIS1=64, NAXIS2=64, NAXIS3=1, NAXIS4=1,
            CTYPE1='RA---TAN', CTYPE2='DEC--TAN', CTYPE3='FREQ', CTYPE4='STOKES',
            CUNIT1='deg', CUNIT2='deg', CUNIT3='Hz',
            CDELT1=-10/3600000, CDELT2=10/3600000, CDELT3=1, CDELT4=1,
            CRPIX1=32.5, CRPIX2=32.5, CRPIX3=1, CRPIX4=1,
            CRVAL1=130, CRVAL2=-44, CRVAL3=3.5e11, CRVAL4=1,
            BUNIT='Jy/beam', BMAJ=40/3600000, BMIN=30/3600000, BPA=20,
        ))
        return image, header

    def validate_alma(self, image, header):
        return load_alma_observation('/data', 'alma.fits',
                                     lambda *args: (image, header, 10., 64),
                                     alma.alma_image_spec)

    def test_alma_preserves_image_values(self):
        image, header = self.alma_fixture()
        image[31, 31] = np.nan  # Existing scorer masks nonfinite observation pixels.
        original = image.copy()
        result = self.validate_alma(image, header)
        self.assertIs(result[0], image)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(len(result), 5)

    def test_alma_invalid_metadata(self):
        for fault in ('shape', 'beam', 'frequency', 'units', 'wcs', 'centre'):
            image, header = self.alma_fixture()
            if fault == 'shape':
                image = image[:-1]
            elif fault == 'beam':
                header['BMIN'] = 0
            elif fault == 'frequency':
                header['CRVAL3'] = -1
            elif fault == 'units':
                header['BUNIT'] = 'mJy/beam'
            elif fault == 'wcs':
                header['CTYPE1'] = 'LINEAR'
            else:
                header['CRPIX1'] = 1000
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, 'ALMA.*alma.fits') as caught:
                self.validate_alma(image, header)
            self.assertIsNotNone(caught.exception.__cause__)

    def test_alma_metadata_validation_does_not_judge_background_snr_or_aperture(self):
        for case in ('background', 'no_detection', 'aperture', 'zero', 'nan', 'inf'):
            image, header = self.alma_fixture()
            if case == 'background':
                radius, *_ = pdi.compute_grid(image)
                image[radius > 15] = np.nan
            elif case == 'no_detection':
                image[image == 1] = .01
            elif case == 'aperture':
                image[0, 0] = 1
            else:
                image[:] = {'zero': 0, 'nan': np.nan, 'inf': np.inf}[case]
            with self.subTest(case=case), patch.object(alma, 'alma_snr_aperture_radius') as selector:
                result = self.validate_alma(image, header)
                self.assertIs(result[0], image)
                selector.assert_not_called()

    def test_alma_scoring_uses_precomputed_rms_and_aperture(self):
        image, header = self.alma_fixture()
        spec = self.validate_alma(image, header)[4]
        data = dict(alma_cont=image, header=header, ps_alma=10.,
                    alma_wavelength=spec['wavelength_um'], image_spec=spec,
                    noise_level_alma=None, aperture_radius_mas=None)
        radius, *_ = pdi.compute_grid(image)
        expected_noise = np.nanstd(image[radius > 15])
        expected_aperture = 10 * alma.alma_snr_aperture_radius(
            image, expected_noise, center_xy=(31.5, 31.5))
        data['noise_level_alma'] = expected_noise
        data['aperture_radius_mas'] = expected_aperture
        model = .9 * image
        expected = alma.compare_alma_images(
            image, model, header, 10., center_xy=(31.5, 31.5),
            aperture_radius_mas=expected_aperture, noise_rms_jybeam=expected_noise)
        compare = alma.compare_alma_images

        def compare_without_plot(*args, **kwargs):
            kwargs['save_path'] = None
            return compare(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(alma, 'load_mcfost_image_alma_casa', return_value=(None, header, model, None)), \
                patch.object(alma, 'convolve_alma_to_jybeam', return_value=(model, 10.)), \
                patch.object(alma, 'align_alma_to_observation', return_value=(model, {'observed_center_xy': (31.5, 31.5)})), \
                patch.object(alma, 'alma_snr_aperture_radius', side_effect=AssertionError('Aperture recalculated during scoring')), \
                patch.object(alma, 'compare_alma_images', side_effect=compare_without_plot) as comparison:
            work = Path(directory)
            (work / 'data_th').mkdir()
            (work / 'data_th/sed_rt.fits.gz').touch()
            loss, info = mcfost.load_and_score_outputs(
                {'stage': 'alma', 'products': ['alma']}, work, {'alma': data},
                SimpleNamespace(plot_intermediate=False), {})
        self.assertEqual(comparison.call_args.kwargs['noise_rms_jybeam'], expected_noise)
        self.assertEqual(comparison.call_args.kwargs['aperture_radius_mas'], expected_aperture)
        self.assertEqual(loss, expected['residual_energy_loss'])
        for key, value in expected.items():
            np.testing.assert_equal(info['alma']['comparison'][key], value, err_msg=key)
        self.assertEqual(data['noise_level_alma'], expected_noise)
        self.assertEqual(data['aperture_radius_mas'], expected_aperture)

    def test_alma_reader_error(self):
        original = FileNotFoundError('file missing')
        with self.assertRaisesRegex(ValueError, 'ALMA.*alma.fits') as caught:
            load_alma_observation('/data', 'alma.fits', Mock(side_effect=original), Mock())
        self.assertIs(caught.exception.__cause__, original)

    def test_interferometry_validation_has_source_and_original_cause(self):
        original = ValueError('no retained observations')
        reader = Mock(return_value=SimpleNamespace(vis_in_fcorr=False))
        with self.assertRaisesRegex(RuntimeError, 'vis2_chromatic.*pionier.*PIONIER/.*fits') as caught:
            load_interferometry_data('demo_ozstar', ['vis2_chromatic'], reader,
                                     validator=Mock(side_effect=original))
        self.assertIs(caught.exception.__cause__, original)
        self.assertEqual(reader.call_count, 1)

    def test_shared_loader_validates_only_requested_products_before_profiles(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'SMAC3.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'load_data')
        obs = SimpleNamespace(load_sed_data=Mock(return_value=([1, 2], [3, 4], [.1, .1])))
        polarimetry = SimpleNamespace(Loadimage=Mock(side_effect=OSError('missing PDI')),
                                     profiles=Mock(), plot_polarimetric_image=Mock(),
                                     compute_grid=pdi.compute_grid)
        alma_module = SimpleNamespace(Loadimage_alma=Mock(side_effect=OSError('missing ALMA')),
                                     alma_image_spec=alma.alma_image_spec,
                                     alma_snr_aperture_radius=Mock(wraps=alma.alma_snr_aperture_radius))
        reader = Mock(side_effect=OSError('missing interferometry'))
        namespace = dict(Path=Path, np=np, obs=obs, obp=polarimetry, oba=alma_module,
                         load_interferometry_data=load_interferometry_data,
                         distroi=SimpleNamespace(read_oi_container_from_oifits=reader),
                         obi=SimpleNamespace(validate_interferometric_data=Mock()))
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'SMAC3.py', 'exec',
                     flags=__future__.annotations.compiler_flag), namespace)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            loader = namespace['load_data']
            loader('demo_ozstar', directory, ['sed'])
            polarimetry.Loadimage.assert_not_called()
            alma_module.Loadimage_alma.assert_not_called()
            alma_module.alma_snr_aperture_radius.assert_not_called()
            reader.assert_not_called()
            # ALMA is required at startup even when only a later stage requests it.
            products = fidelity_products({'stages': [{'products': ['sed']}, {'products': ['sed', 'alma']}]})
            with self.assertRaisesRegex(ValueError, 'ALMA.*fits'):
                loader('demo_ozstar', directory, products)
            with self.assertRaisesRegex(ValueError, 'PDI V.*fits'):
                loader('demo_ozstar', directory, ['pdi_V'])
            polarimetry.Loadimage.side_effect = None
            polarimetry.Loadimage.return_value = np.ones((220, 220))
            # The legacy preset has only PI, so cannot supply the requested score.
            with self.assertRaisesRegex(ValueError, 'PDI V.*components'):
                loader('demo_mac', directory, ['pdi_V'])
            polarimetry.profiles.assert_not_called()
            image, header = self.alma_fixture()
            alma_module.Loadimage_alma.side_effect = None
            alma_module.Loadimage_alma.return_value = (image, header, 10., 64)
            polarimetry.profiles.return_value = (None, None)
            data = loader('demo_ozstar', directory, ['alma'])
            self.assertIs(data['alma']['alma_cont'], image)
            radius, *_ = pdi.compute_grid(image)
            noise = np.nanstd(image[radius > 15])
            expected_aperture = 10 * alma.alma_snr_aperture_radius(
                image, noise, center_xy=(31.5, 31.5))
            self.assertEqual(data['alma']['noise_level_alma'], noise)
            self.assertEqual(data['alma']['aperture_radius_mas'], expected_aperture)
            self.assertEqual(alma_module.alma_snr_aperture_radius.call_count, 1)


if __name__ == '__main__':
    unittest.main()
