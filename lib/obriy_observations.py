"""Validate required observations before profiles, workers or simulations."""
from pathlib import Path

import numpy as np


def load_sed_observation(path, reader):
    """Load and validate a spectral energy distribution (SED) observation. Wrapper on top of previous obs.load_sed_data"""
    try:
        columns = tuple(np.asarray(column, dtype=float) for column in reader(path))
        if len(columns) != 3 or any(column.ndim != 1 for column in columns):
            raise ValueError("Expected 1D wavelength, flux and uncertainty arrays.")
        wave, flux, error = columns
        if wave.size < 2 or flux.shape != wave.shape or error.shape != wave.shape:
            raise ValueError("SED requires at least two points and matching array lengths.")
        if not all(np.isfinite(column).all() for column in columns):
            raise ValueError("SED wavelengths, fluxes and uncertainties must be finite.")
        if np.any(wave <= 0) or np.any(error <= 0):
            raise ValueError("SED wavelengths and uncertainties must be positive.")
        return columns
    except Exception as error:
        raise ValueError(f"Invalid or missing SED observations at {path}: {error}") from error


def _image(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("Expected a nonempty 2D image.")
    return array


def _require_pdi_image(image):
    """Require an image payload without inspecting its pixel values."""
    if image is None or np.ndim(image) != 2 or 0 in np.shape(image):
        raise ValueError("Required PDI image is missing or empty (expected a 2D image).")


def load_pdi_image(folder, filename, reader, label):
    path = Path(folder) / filename
    try:
        image = reader(folder, filename)
        _require_pdi_image(image)
        return image
    except Exception as error:
        raise ValueError(f"Invalid or missing {label} at {path}: {error}") from error


def pdi_scoring_aperture(shape, pixel_scale_mas, radius_mas=500.0):
    """Use the scorer's geometric centre and circular pixel-edge boundary."""
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("PDI requires a nonempty 2D grid.")
    if not np.all(np.isfinite([pixel_scale_mas, radius_mas])) or min(pixel_scale_mas, radius_mas) <= 0:
        raise ValueError("PDI pixel scale and aperture radius must be positive and finite.")
    ny, nx = shape
    yy, xx = np.indices(shape)
    radius = np.hypot(xx - (nx - 1) / 2, yy - (ny - 1) / 2) * pixel_scale_mas
    outer = min(float(radius_mas), min(nx, ny) * pixel_scale_mas / 2)
    return radius, outer, radius <= outer


def validate_pdi_observation(images, psf, pixel_scale_mas, band, sources):
    """Require compatible images, finite aperture samples and nonnegative sum_I.

    Individual NaNs are allowed. The intensity sum ignores NaNs, but must
    remain finite and nonnegative; the observation arrays are not modified.
    """
    try:
        required = ("I", "PI", "Q_phi", "U_phi", "Q", "U")
        if not isinstance(images, dict) or any(key not in images for key in required):
            raise ValueError(f"PDI requires image components {', '.join(required)}.")
        for key in required:
            _require_pdi_image(images[key])
        _require_pdi_image(psf)
        arrays = {key: np.asarray(images[key], dtype=float) for key in required}
        shape = arrays["I"].shape
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("PDI components must share the same 2D pixel grid.")
        _, _, aperture = pdi_scoring_aperture(shape, pixel_scale_mas)
        for key, array in arrays.items():
            if not np.isfinite(array[aperture]).any():
                raise ValueError(f"{key} has no finite samples in the scoring aperture.")
        with np.errstate(over="ignore", invalid="ignore"):
            sum_i = np.nansum(arrays["I"][aperture])
        if not np.isfinite(sum_i) or sum_i <= 0:
            raise ValueError("PDI aperture sum_I must be finite and nonnegative.")
    except Exception as error:
        paths = ", ".join(f"{key}={path}" for key, path in sources.items())
        raise ValueError(f"Invalid PDI {band} observations ({paths}): {error}") from error


def load_alma_observation(folder, filename, reader, image_spec):
    """Load the image and validate metadata, without judging pixel values.

    The shared load_data() separately prepares RMS and aperture for scoring.
    """
    from astropy import units as u
    from astropy.wcs import WCS

    path = Path(folder) / filename
    try:
        image, header, pixel_scale, image_size = reader(folder, filename)
        array = _image(image)
        if array.shape != (int(header["NAXIS2"]), int(header["NAXIS1"])):
            raise ValueError("ALMA image dimensions disagree with the FITS header.")
        for key in ("CDELT1", "CDELT2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "BMAJ", "BMIN", "BPA"):
            if not np.isfinite(float(header[key])):
                raise ValueError(f"ALMA metadata {key} must be finite.")
        if u.Unit(header["BUNIT"]) != u.Jy / u.beam:
            raise ValueError("ALMA data must be in Jy/beam; no automatic rescaling is applied.")
        celestial = WCS(header).celestial
        if not celestial.has_celestial or not np.isfinite(celestial.pixel_scale_matrix).all():
            raise ValueError("ALMA requires a valid celestial WCS.")
        if np.linalg.det(celestial.pixel_scale_matrix) == 0:
            raise ValueError("ALMA celestial WCS is singular.")
        if not np.isfinite(pixel_scale) or pixel_scale <= 0:
            raise ValueError("ALMA pixel scale must be positive and finite.")
        for axis in (1, 2):
            scale = (abs(float(header[f"CDELT{axis}"])) * u.Unit(header[f"CUNIT{axis}"])).to_value(u.mas)
            if not np.isclose(scale, pixel_scale):
                raise ValueError("ALMA requires matching square pixel scales in mas.")
        if min(float(header["BMAJ"]), float(header["BMIN"])) <= 0:
            raise ValueError("ALMA beam axes must be positive.")
        spec = image_spec(header)
        ny, nx = array.shape
        centre = (float(header["CRPIX1"]) - 1, float(header["CRPIX2"]) - 1)
        if not (0 <= centre[0] < nx and 0 <= centre[1] < ny):
            raise ValueError("ALMA reference pixel must lie inside the image.")
        return image, header, pixel_scale, image_size, spec
    except Exception as error:
        raise ValueError(f"Invalid or missing ALMA observations at {path}: {error}") from error
