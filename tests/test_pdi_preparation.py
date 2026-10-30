"""Reuse fixed observation profiles while recalculating trial-dependent quantities."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from ConfigSpace import ConfigurationSpace, Constant, Float

import SMAC3
from lib import obriy_polarimetry as pdi


def observation_images(shape):
    yy, xx = np.indices(shape, dtype=float)
    xx -= (shape[1] - 1) / 2
    yy -= (shape[0] - 1) / 2
    phi = np.arctan2(yy, xx)
    intensity = 2 + np.exp(-(xx**2 + yy**2) / 50.)
    qphi = .1 * intensity * (1 + .4 * np.cos(2 * phi + .3))
    uphi = .02 * intensity * np.sin(phi)
    return dict(I=intensity, Q_phi=qphi, U_phi=uphi)


class PDIPreparationTests(unittest.TestCase):
    def test_fixed_configuration_overrides_template_distance(self):
        template = dict(grid_nx='1000', grid_ny='1000', grid_size='800.', distance_pc='1000.')
        configspace = ConfigurationSpace()
        configspace.add(Constant('distance_pc', 1220.))
        with patch.object(SMAC3.obm, 'ParaFile', return_value=SimpleNamespace(params=template)) as reader:
            grid = SMAC3.fixed_pdi_grid(Path('/run/simulation.para'), configspace)
        reader.assert_called_once_with('/run/simulation.para')
        self.assertEqual(grid['distance_pc'], 1220.)
        self.assertEqual(grid['grid_size'], '800.')
        self.assertEqual(template['distance_pc'], '1000.')
        variable_grid = ConfigurationSpace()
        variable_grid.add(Float('distance_pc', (1000., 1500.)))
        with patch.object(SMAC3.obm, 'ParaFile', return_value=SimpleNamespace(params=template)):
            with self.assertRaisesRegex(ValueError, 'distance_pc must be fixed'):
                SMAC3.fixed_pdi_grid('/run/simulation.para', variable_grid)

    def test_template_radius_matches_actual_resampling(self):
        observation = observation_images((220, 220))
        for nx, ny, size_au in ((1000, 1000, 800.), (80, 60, 400.), (64, 64, 2000.)):
            for pixel_scale in (3.6, 12.27):
                with self.subTest(nx=nx, ny=ny, scale=pixel_scale), redirect_stdout(io.StringIO()):
                    grid = dict(grid_nx=nx, grid_ny=ny, grid_size=size_au, distance_pc=1220.)
                    prepared = pdi.prepare_pdi_observation(observation, pixel_scale, grid)
                    native_pixel = 1000 * size_au / 1220 / max(nx, ny)
                    native = np.ones((ny, nx))
                    # The PDI rescaling helper uses this image resampler before
                    # calculating Stokes-derived quantities (which require square images).
                    resampled = pdi.rescale(native, native_pixel / pixel_scale, channel_axis=None)
                    shape = resampled.shape
                    self.assertEqual(prepared['model_shape'], shape)
                    radius = min(500., 220 * pixel_scale / 2, min(shape) * pixel_scale / 2)
                    self.assertEqual(prepared['profiles']['aperture_radius_mas'], radius)

    def test_profiles_are_reused_and_quadrants_follow_each_trial_pa(self):
        observation = observation_images((48, 48))
        grid = dict(grid_nx=32, grid_ny=32, grid_size=115.2, distance_pc=1000.)
        prepared = pdi.prepare_pdi_observation(observation, 3.6, grid, radial_bin_mas=20.)
        # Worker inputs must survive spawn/pickling without changes.
        prepared = pickle.loads(pickle.dumps(prepared))
        before = pickle.dumps(prepared)
        model_observation = observation_images((32, 32))
        _, _, _, phi = pdi.compute_qphi_uphi_pi(model_observation['Q_phi'], model_observation['U_phi'])
        qp = model_observation['Q_phi']
        up = model_observation['U_phi']
        model = dict(q_phi=qp, img_tot=model_observation['I'],
                     img_q=-qp*np.cos(2*phi)+up*np.sin(2*phi),
                     img_u=-qp*np.sin(2*phi)-up*np.cos(2*phi))
        previous_quadrants = None
        for angle in (0., 37., 100.):
            expected = pdi.compare_pdi_constraints(
                observation, model, 3.6, radial_bin_mas=20., disk_pa_deg=angle)
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(pdi, 'measure_pdi_profiles', wraps=pdi.measure_pdi_profiles) as profiles, \
                    patch.object(pdi, 'measure_pdi_azimuthal_profile', wraps=pdi.measure_pdi_azimuthal_profile) as azimuthal, \
                    patch.object(pdi, 'plot_pdi_constraints'), \
                    patch.object(pdi, 'plot_differential_quadrants_slides'):
                actual = pdi.compare_pdi_constraints(
                    observation, model, 3.6, radial_bin_mas=20., disk_pa_deg=angle,
                    prepared_observation=prepared, output_path=Path(directory) / 'pdi.png')
                # Only model profiles are recalculated, including when plotting.
                self.assertEqual(profiles.call_count, 1)
                self.assertIs(profiles.call_args.args[0]['Q_phi'], model['q_phi'])
                self.assertEqual(azimuthal.call_count, 1)
                self.assertIs(azimuthal.call_args.args[0], model['q_phi'])
                self.assertTrue((Path(directory) / 'pdi_azimuthal.png').is_file())
            self.assertEqual(actual, expected)
            if previous_quadrants is not None:
                self.assertFalse(np.allclose(actual['observed']['quadrants'], previous_quadrants))
            previous_quadrants = actual['observed']['quadrants']
        self.assertEqual(pickle.dumps(prepared), before)
        self.assertNotIn('quadrants', prepared['profiles'])

    def test_changed_grid_or_bins_cannot_use_stale_observation_profiles(self):
        observation = observation_images((32, 32))
        grid = dict(grid_nx=32, grid_ny=32, grid_size=115.2, distance_pc=1000.)
        prepared = pdi.prepare_pdi_observation(observation, 3.6, grid)
        model = dict(q_phi=observation['Q_phi'], img_tot=observation['I'],
                     img_q=prepared['images']['Q'], img_u=prepared['images']['U'])
        for change in ('shape', 'scale', 'bins'):
            with self.subTest(change=change):
                changed_model = {key: image[:30, :30] for key, image in model.items()} if change == 'shape' else model
                with self.assertRaisesRegex(ValueError, 'differ from the prepared observations'):
                    pdi.compare_pdi_constraints(
                        observation, changed_model, 4. if change == 'scale' else 3.6,
                        radial_bin_mas=30. if change == 'bins' else 25., prepared_observation=prepared)


if __name__ == '__main__':
    unittest.main()
