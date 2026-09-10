import os
import fnmatch
from astropy.io import fits
from pathlib import Path
import numpy as np
from typing import Literal, Tuple, Dict, Optional, Union, Any, List
import subprocess

from astropy import units as u
from astropy.convolution import convolve_fft
from astropy.wcs import WCS
from radio_beam import Beam


#from distroi.auxiliary import constants
from distroi.data import image
from distroi.data import sed
from distroi.model.geom_comp import geom_comp
from distroi.auxiliary import select_data_oifits
from distroi.data.oi_container import OIContainer

import matplotlib.pyplot as plt


from skimage.transform import rescale


import lib.obriy_general as obg
import lib.obriy_interferometry as obi
import lib.obriy_sed as obs
import lib.obriy_mcfost as obm
import lib.obriy_polarimetry as obp

from astropy import units as u
from astropy.constants import c


def alma_image_spec(header):
    """
    Image requirements for a single-channel ALMA continuum observation.
    """
    frequency_axes = [
        axis
        for axis in range(1, int(header["NAXIS"]) + 1)
        if str(header.get(f"CTYPE{axis}", "")).upper().startswith("FREQ")
    ]
    if len(frequency_axes) != 1:
        raise ValueError("Expected one FITS frequency axis.")

    axis = frequency_axes[0]
    if int(header[f"NAXIS{axis}"]) != 1:
        raise ValueError("Expected a single-channel continuum image.")

    # Frequency at the sole channel, whose FITS pixel coordinate is 1.
    frequency_value = (
        float(header[f"CRVAL{axis}"])
        + (1.0 - float(header[f"CRPIX{axis}"]))
        * float(header[f"CDELT{axis}"])
    )
    frequency = frequency_value * u.Unit(header[f"CUNIT{axis}"])
    frequency_hz = frequency.to_value(u.Hz)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Invalid ALMA observing frequency.")

    wavelength_um = (c / frequency).to_value(u.um)

    pixel_x_mas = (
        abs(float(header["CDELT1"])) * u.Unit(header["CUNIT1"])
    ).to_value(u.mas)
    pixel_y_mas = (
        abs(float(header["CDELT2"])) * u.Unit(header["CUNIT2"])
    ).to_value(u.mas)

    if not np.isclose(pixel_x_mas, pixel_y_mas):
        raise ValueError("This image setup expects square observation pixels.")

    beam_major_mas = (float(header["BMAJ"]) * u.deg).to_value(u.mas)
    beam_minor_mas = (float(header["BMIN"]) * u.deg).to_value(u.mas)
    if min(pixel_x_mas, beam_major_mas, beam_minor_mas) <= 0:
        raise ValueError("Pixel and beam sizes must be positive.")

    # Cover both edges relative to the reference pixel, including pixel edges.
    # Assumes that the model centre will be aligned with this reference pixel.
    half_width_x = max(
        float(header["CRPIX1"]) - 0.5,
        float(header["NAXIS1"]) + 0.5 - float(header["CRPIX1"]),
    ) * pixel_x_mas
    half_width_y = max(
        float(header["CRPIX2"]) - 0.5,
        float(header["NAXIS2"]) + 0.5 - float(header["CRPIX2"]),
    ) * pixel_y_mas

    # Five Gaussian sigma on every side for subsequent beam convolution.
    beam_sigma_mas = beam_major_mas / np.sqrt(8.0 * np.log(2.0))
    padding_mas = 5.0 * beam_sigma_mas

    # Sample at least twice as finely as the observations and
    # with at least eight pixels across the minor beam FWHM.
    model_pixel_mas = min(pixel_x_mas / 2.0, beam_minor_mas / 8.0)
    required_fov_mas = 2.0 * (
        max(half_width_x, half_width_y) + padding_mas
    )

    npix = int(np.ceil(required_fov_mas / model_pixel_mas))
    if npix % 2:
        npix += 1

    return {
        "frequency_hz": float(frequency_hz),
        "wavelength_um": float(wavelength_um),
        "npix": npix,
        "pixel_mas": float(model_pixel_mas),
        "fov_mas": float(npix * model_pixel_mas),
        "beam_major_mas": float(beam_major_mas),
        "beam_minor_mas": float(beam_minor_mas),
        "beam_pa_deg": float(header["BPA"])
    }


def Loadimage_alma(dirdat,filename):
    """
    Loading reduced data fits from ALMA

    Parameters:
    dirdat: str
        Path
    filename: str
        Filename or part of it

    """
    dir =dirdat
    psfile =  filename
    files = os.listdir(dir)
    for file in files:
        if fnmatch.fnmatch(file, psfile):
            hdulPSF = fits.open(dir + file)
            fit = hdulPSF[0].data
            header = hdulPSF[0].header
            data=fit[0,0,:,:]
            pix_scale=header["CDELT2"]*3600*1000 #deg to mas
            data_size=header["NAXIS1"] #pixels

            
    return data, header, pix_scale, data_size





def load_mcfost_image_alma_casa(
        main_dir: str,
        wavelength: str, 
        *,
        ploting: bool= False, 
        save_plots: str = None, 
        title_addition:str = ''
) -> tuple[np.ndarray, fits.Header, np.ndarray, float]:
        """
        Load and optionally visualize MCFOST model images with "casa" option in mcfost run.

        This function:
        * Locates the folder corresponding to a given wavelength (``main_dir/data_<wavelength>``),
        * Ensures the FITS file is uncompressed (unzips ``RT.fits.gz`` if only that exists),
        * Loads the multi-extension FITS file produced by MCFOST,
        * Extracts key image components: total intensity, Stokes parameters (Q, U, V),
            direct stellar emission, scattered stellar emission, thermal emission, and scattered
            thermal emission,
        * Optionally produces a quick-look decomposition plot.

        Parameters
        ----------
        main_dir : str
            Path to the parent directory containing the MCFOST run output.
        wavelength : str or float
            Wavelength identifier used by MCFOST in the subdirectory name (e.g., ``1.65`` or ``1650``).
        ploting : bool, default=False
            If True, generate a figure showing total intensity, Stokes components, and decomposed images.
        save_plots : bool or str, default=False
            If False, plots are not saved. If a string (directory path), plots are saved into this directory
            with filenames including the wavelength and `title_addition`.
        title_addition : str, optional
            Additional string to append to the figure title and saved filename.

        Returns
        -------
        img_array : np.ndarray
            Raw 5D image data array from the FITS file (contains all components).
        header_data : astropy.io.fits.Header
            Header from the FITS file containing metadata (including wavelength).
        img_tot : np.ndarray
            Total intensity image (I).
        pix_scale : float
            Pixel scale in milliarcseconds (mas), derived from the FITS header.
     

        Notes
        -----
        * The function assumes that the MCFOST output directory structure is of the form:
        ``<main_dir>/data_<wavelength>/RT.fits(.gz)``.
        * Image arrays are returned in pixel coordinates, not rescaled to angular size.
        * The optional plots use a fixed colormap ("afmhot") and a 2×4 grid layout.
        """
        folderpath=main_dir+'/data_'+str(wavelength)
    
        if os.path.exists(folderpath+'/RT.fits.gz')and not os.path.exists(folderpath+'/RT.fits'):
            #os.system("gunzip -k "+folderpath+'/RT.fits.gz')
            subprocess.run(
                    ["gunzip", "-k", f"{folderpath}/RT.fits.gz"],
                    stdout=subprocess.DEVNULL
                )
        #open the required fits file + get some header info
        hdul=fits.open(folderpath+'/RT.fits')
        header_data=hdul[0].header

        wave=hdul[0].header['WAVE']
        pix_scale=hdul[0].header['CDELT2']*3600*1000 #deg to mas
        #load in all images and separate 
        img_array=hdul[0].data
        #total intensity
        img_tot=img_array[:][:][0][0]
        
        
        if ploting==True:
            #do some plotting
            fig, ax = plt.subplots(1, 1, figsize=(7,7))
            color_map = 'viridis' #'afmhot'
            ax.imshow(img_tot, color_map, extent=[+img_tot.shape[0]/2, -img_tot.shape[0]/2, -img_tot.shape[1]/2, img_tot.shape[1]/2])
            ax.set_title("I$_{tot}$")
            plt.suptitle(str(wave)+"$\mu m$, "+title_addition)
            #plt.tight_layout()
            #plt.show()
            #save the plots
            if save_plots:
                fig.savefig(save_plots+title_addition+' image_decomp_'+str(wave)+'.png', dpi= 150, bbox_inches='tight')
            plt.close(fig)
        return img_array, header_data, img_tot, pix_scale


def rescale_alma(
    img_tot: np.ndarray,
    current_pix_scale: float,
    new_pix_scale: float,
    *,
    conserve: str = "surface_brightness",  # or "sum"
    order: int = 3,                        # cubic interpolation
) -> Tuple[np.ndarray]:
    """
    Resample image to a new pixel scale.

    Parameters
    ----------
    current_pix_scale : float
        Current pixel scale (e.g., mas/pix).
    instrument : {'irdis','zimpol', None}
        If provided and new_pix_scale is None, use instrument default pixel scale.
    new_pix_scale : float, optional
        Target pixel scale (same units as current_pix_scale).
    conserve : {'surface_brightness','sum'}
        - 'surface_brightness': keep per-area units unchanged (no intensity renorm).
        - 'sum': renormalize intensities by (current/new)^2 to conserve total flux.
    order : int
        Interpolation order passed to skimage.transform.rescale (0..5).

    Returns
    -------
    img_tot_res
    """
   
    if current_pix_scale <= 0 or new_pix_scale <= 0:
        raise ValueError("Pixel scales must be positive.")

    # scale factor for image dimensions
    s = current_pix_scale / new_pix_scale  # >1 means upsample (finer pixels)

    # Rescale with photometry preserved (no auto [0,1] normalization)
    def _res(x):
        return rescale(
            np.asarray(x, dtype=float),  # ensure float for interpolation
            scale=s,
            order=order,
            anti_aliasing=(s < 1.0),     # AA only when downsampling
            preserve_range=True,
            channel_axis=None,
        )


    img_tot_res = _res(img_tot)

    # Optional intensity renormalization to conserve total flux
    if conserve == "sum":

        img_tot_res /= s**2
    elif conserve != "surface_brightness":
        raise ValueError("`conserve` must be 'surface_brightness' or 'sum'.")


    return img_tot_res


def cut_down_alma(
    img_1: np.ndarray,
    img_2: np.ndarray
) -> Tuple[np.ndarray]:
    """
    Cut down the  image1 to match the size of the  image2.

    Parameters
    ----------
    img_1 : np.ndarray
        First image.
    img_2 : np.ndarray
        Second image.
    
    Returns
    -------
    img_tot_res
    """
    size1 = img_1.shape[0]
    size2= img_2.shape[0]
    try:
        if size1 <= 0 or size2 <= 0:
            raise ValueError("Image dimensions must be positive.")
        if size1 < size2:
            raise ValueError("First image must be larger than second image. Size of first image: {}, size of second image: {}".format(size1, size2))
        diff = (size1 - size2)
        # For an odd difference, the extra pixel is removed from the bottom and right side.
        start= diff// 2
        end = start + size2
        if diff % 2 != 0:
            print("For an odd difference, the extra pixel is removed from the bottom and right side. Possible geometrical shift! Size of first image: {}, size of second image: {}".format(size1, size2))
        img_tot_res = img_1[start:end, start:end]
    except ValueError as e:
        print(f"Error in cut_down_alma: {e}. Returning the original simulated image.")
        img_tot_res = img_1

    return img_tot_res


def chi2_ALMA(main_dir, data_alma, model_jybeam, plot=False, fig_dir=None, extra_title=""):
    """
    Compute the chi2 for ALMA data.
    """
    # Output folder
    if fig_dir is None:
            fig_dir = str(main_dir +"/ALMA")
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    #alma_cont = data_alma['alma_cont']
    obs_rad_prof = data_alma['radial_profile']
    obs_az_prof = data_alma['azimuthal_profile']
    ps_alma = data_alma['ps_alma']
    alma_spec = data_alma['image_spec']
    wavelength_text = f'{alma_spec["wavelength_um"]:.3f}'

    # Load the simulation data for ALMA
    simulated_itot = np.asarray(model_jybeam, dtype=float)

    if simulated_itot.shape != data_alma["alma_cont"].shape:
        raise ValueError("Model and observed image shapes differ.")
    #compute profiles
    if plot:
        obp.plot_polarimetric_image(simulated_itot, ps_alma, title=f'Model Itot, alma_cont', save=str(fig_dir)+'/model_itot_alma.png', image_scale='asinh', roi_half_size=100)
          
    radial_profile_alma_model, azimuthal_profile_alma_model = obp.profiles(simulated_itot, ps_alma, 
                                                profile_type="both",
                                                mode="mean",
                                                radial_limit_mas=100,
                                                plot=plot,
                                                save_prefix=str(fig_dir)+extra_title+'_profile_',
                                                deprojection_inc_pa_deg=(0.0, 0.0),
                                                center=None,
                                                az_nbins=18,
                                                azimuthal_r_in_mas=0.0,
                                                azimuthal_r_out_mas=100.0,
                                                theta0=0.0
                                                )
    
    profile_rad_pi_chi2, _,_, profile_rad_pi_npoints = obp.profile_chi2(obs_rad_prof, radial_profile_alma_model, ps_alma, profile_type="radial", plot=plot, save_prefix=str(fig_dir)+extra_title+'_radial_profile_')
    profile_az_pi_chi2, _,_, profile_az_pi_npoints = obp.profile_chi2(obs_az_prof, azimuthal_profile_alma_model, ps_alma, profile_type="azimuthal", plot=plot, save_prefix=str(fig_dir)+extra_title+'_azimuthal_profile_')
    profiles_chi2_red= (profile_rad_pi_chi2 + profile_az_pi_chi2) / (profile_rad_pi_npoints + profile_az_pi_npoints -2)
            


    return profiles_chi2_red, profile_rad_pi_chi2, profile_az_pi_chi2, profile_rad_pi_npoints, profile_az_pi_npoints


def convolve_alma_to_jybeam(model_jypixel, model_header, observed_header):
    """
    Convolve an intrinsic MCFOST Jy/pixel image with the observed beam.

    Returns
    -------
    model_jybeam : ndarray
        Convolved image on the original model grid, in Jy/beam.
    model_pixel_mas : float
        Native model pixel scale.

    Supports square, unrotated pixels with RA decreasing along columns
    and Dec increasing along rows, as in the supplied MCFOST header.
    """
    model = np.asarray(model_jypixel, dtype=float)

    if model.ndim != 2 or not np.all(np.isfinite(model)):
        raise ValueError("Expected a finite 2D model image.")

    expected_shape = (
        int(model_header["NAXIS2"]),
        int(model_header["NAXIS1"]),
    )
    if model.shape != expected_shape:
        raise ValueError("Model array shape does not match its FITS header.")

    def normalized_unit(header):
        return str(header.get("BUNIT", "")).replace(" ", "").upper()

    if normalized_unit(model_header) != "JY/PIXEL":
        raise ValueError("The native model must have BUNIT=JY/PIXEL.")

    if normalized_unit(observed_header) != "JY/BEAM":
        raise ValueError("The observation must have BUNIT=Jy/beam.")

    # This helper applies a full restoring beam to an intrinsic image.
    # Avoid accidentally convolving an already restored image twice.
    if any(key in model_header for key in ("BMAJ", "BMIN")):
        raise ValueError("Model contains beam metadata; inspect before convolving.")

    for key in ("BMAJ", "BMIN", "BPA"):
        if key not in observed_header:
            raise ValueError(f"Observation is missing {key}.")

    beam = Beam.from_fits_header(observed_header)
    beam_values = [
        beam.major.to_value(u.deg),
        beam.minor.to_value(u.deg),
        beam.pa.to_value(u.deg),
    ]
    if (
        not np.all(np.isfinite(beam_values))
        or not beam_values[0] >= beam_values[1] > 0
    ):
        raise ValueError("Invalid observed restoring beam.")

    # Astropy incorporates CDELT and pixel coord/coordinate description into this matrix.
    # For celestial WCS (world coordinate system) its angular units are degrees.
    model_wcs = WCS(model_header).celestial
    if not (
        model_wcs.wcs.ctype[0].startswith("RA")
        and model_wcs.wcs.ctype[1].startswith("DEC")
    ):
        raise ValueError("Expected RA, Dec spatial axes.")

    matrix = np.asarray(model_wcs.pixel_scale_matrix, dtype=float)
    dx_deg = matrix[0, 0]
    dy_deg = matrix[1, 1]

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Invalid spatial WCS.")

    if not np.allclose(
        matrix,
        np.diag([dx_deg, dy_deg]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Rotated or skewed model grids need a WCS-aware kernel.")

    if not (
        dx_deg < 0 < dy_deg
        and np.isclose(abs(dx_deg), dy_deg, rtol=1e-6, atol=0.0)
    ):
        raise ValueError(
            "Expected square pixels, RA decreasing and Dec increasing."
        )

    pixel_scale = abs(dx_deg) * u.deg
    pixel_area = abs(np.linalg.det(matrix)) * u.deg**2

    # Explicit support consistent with the five-sigma image padding.
    sigma_major_pixels = (
        beam.major / pixel_scale
    ).to_value(u.dimensionless_unscaled) / np.sqrt(8.0 * np.log(2.0))
    kernel_size = 2 * int(np.ceil(5.0 * sigma_major_pixels)) + 1

    # as_kernel adds the PA-to-array-axis rotation internally.
    # Do not negate BPA or add another 90 degrees.
    kernel = beam.as_kernel(
        pixel_scale,
        x_size=kernel_size,
        y_size=kernel_size,
        mode="center",
    )

    convolved_jypixel = convolve_fft(
        model,
        kernel,
        normalize_kernel=True,
        boundary="fill",
        fill_value=0.0,
        nan_treatment="fill",
        fft_pad=True,
        psf_pad=True,
        crop=True,
    )

    # Unit-sum convolution retains Jy/native pixel.
    # Convert using native pixel area, before resampling.
    pixels_per_beam = (
        beam.sr / pixel_area.to(u.sr)
    ).to_value(u.dimensionless_unscaled)

    model_jybeam = convolved_jypixel * pixels_per_beam
    return model_jybeam, pixel_scale.to_value(u.mas)


def alma_snr_aperture_radius(observed_jybeam, noise_rms_jybeam, *, center_xy,
                             snr_threshold=3.0, padding_pixels=3.0):
    """Return max(R[data > threshold * RMS]) + padding, in pixels.

    Use the full observed image, before cropping, so the aperture is independent
    of model brightness. All detected pixels contribute, including isolated ones.
    """
    observed = np.asarray(observed_jybeam, dtype=float)
    if observed.ndim != 2:
        raise ValueError("Expected a 2D observed image.")
    if noise_rms_jybeam is None:
        raise ValueError("An observed background RMS is required for an SNR aperture.")
    noise = float(noise_rms_jybeam)
    if not np.all(np.isfinite([noise, snr_threshold, padding_pixels])):
        raise ValueError("RMS, SNR threshold and padding must be finite.")
    if noise <= 0 or snr_threshold <= 0 or padding_pixels < 0:
        raise ValueError("RMS and SNR threshold must be positive; padding nonnegative.")
    xc, yc = map(float, center_xy)
    if not np.all(np.isfinite([xc, yc])):
        raise ValueError("Aperture centre must be finite.")
    detected = np.isfinite(observed) & (observed > snr_threshold * noise)
    if not detected.any():
        raise ValueError("No observed pixels exceed the requested SNR threshold.")
    y, x = np.indices(observed.shape)
    radius_pixels = np.hypot(x - xc, y - yc)
    return float(np.max(radius_pixels[detected]) + padding_pixels)


def compare_alma_images(
    observed_jybeam,
    model_jybeam,
    observed_header,
    pixel_scale_mas,
    *,
    center_xy,
    aperture_radius_mas=None,
    snr_threshold=3.0,
    aperture_padding_pixels=3.0,
    radial_bin_width_mas=None,
    noise_rms_jybeam=None,
    save_path=None,
):
    """
    Diagnostics for ALMA images already matched in beam, units and pixel grid.

    center_xy uses zero-based array coordinates: (column, row).
    No amplitude normalisation or positional adjustment is performed.
    With aperture_radius_mas=None, use max(R[observed > snr_threshold * RMS])
    plus aperture_padding_pixels. Pass the full observation for this selection.
    An explicit radius retains the fixed-aperture behaviour.
    """
    observed = np.asarray(observed_jybeam, dtype=float)
    model = np.asarray(model_jybeam, dtype=float)

    if observed.ndim != 2 or observed.shape != model.shape:
        raise ValueError("Expected matching 2D observed and model images.")
    if not np.all(np.isfinite(model)):
        raise ValueError("Model contains non-finite pixels.")
    if not np.isfinite(pixel_scale_mas) or pixel_scale_mas <= 0:
        raise ValueError("Pixel scale must be positive and finite.")
    automatic_aperture = aperture_radius_mas is None
    if automatic_aperture:
        aperture_radius_mas = pixel_scale_mas * alma_snr_aperture_radius(
            observed, noise_rms_jybeam, center_xy=center_xy,
            snr_threshold=snr_threshold, padding_pixels=aperture_padding_pixels,
        )
    if not np.isfinite(aperture_radius_mas) or aperture_radius_mas <= 0:
        raise ValueError("Pixel scale and aperture radius must be positive.")

    beam = Beam.from_fits_header(observed_header)

    # Approximately one beam per radial bin.
    if radial_bin_width_mas is None:
        radial_bin_width_mas = beam.major.to_value(u.mas)
    if radial_bin_width_mas <= 0:
        raise ValueError("Radial bin width must be positive.")

    ny, nx = observed.shape
    xc, yc = map(float, center_xy)

    # Ensure the entire circular aperture lies inside the image.
    available_radius_mas = min(
        xc + 0.5,
        nx - 0.5 - xc,
        yc + 0.5,
        ny - 0.5 - yc,
    ) * pixel_scale_mas
    if aperture_radius_mas > available_radius_mas:
        raise ValueError("The requested aperture extends outside the image.")

    y, x = np.indices(observed.shape)
    radius_mas = np.hypot(x - xc, y - yc) * pixel_scale_mas

    # Keep faint and negative data inside the circle. Exclude invalid observation
    # pixels from both images; invalid model values are rejected above.
    aperture = (radius_mas < aperture_radius_mas) & np.isfinite(observed)
    if not aperture.any():
        raise ValueError("No finite observed pixels within the aperture.")

    # Positive residual means the model underpredicts the observation.
    residual = observed - model
    observed_energy = float(np.sum(observed[aperture]**2))
    residual_energy = float(np.sum(residual[aperture]**2))
    if not np.isfinite(observed_energy) or observed_energy <= 0:
        raise ValueError("Observed aperture energy must be positive and finite.")
    residual_energy_loss = 100.0 * residual_energy / observed_energy
    if not np.isfinite(residual_energy_loss):
        raise ValueError("ALMA aperture residual-energy loss is not finite.")

    # Jy/beam → integrated Jy over the same fixed aperture.
    pixel_area = (pixel_scale_mas * u.mas)**2
    pixel_to_beam = (
        pixel_area.to(u.sr) / beam.sr
    ).to_value(u.dimensionless_unscaled)

    observed_flux = float(observed[aperture].sum() * pixel_to_beam)
    model_flux = float(model[aperture].sum() * pixel_to_beam)

    # Disjoint annuli. Retain absolute Jy/beam brightness.
    edges = np.arange(
        0.0, aperture_radius_mas, radial_bin_width_mas
    )
    edges = np.append(edges, aperture_radius_mas)

    radii = []
    observed_profile = []
    model_profile = []
    counts = []

    for inner, outer in zip(edges[:-1], edges[1:]):
        annulus = (radius_mas >= inner) & (radius_mas < outer) & aperture
        count = int(annulus.sum())
        if count == 0:
            continue

        radii.append(0.5 * (inner + outer))
        observed_profile.append(float(observed[annulus].mean()))
        model_profile.append(float(model[annulus].mean()))
        counts.append(count)

    result = {
        "residual_energy_loss": residual_energy_loss,
        "observed_energy": observed_energy,
        "residual_energy": residual_energy,
        "aperture_pixel_count": int(aperture.sum()),
        "observed_flux_jy": observed_flux,
        "model_flux_jy": model_flux,
        "flux_difference_jy": model_flux - observed_flux,
        "model_to_observed_flux": (
            model_flux / observed_flux if observed_flux != 0 else None
        ),
        "radius_mas": np.asarray(radii),
        "observed_profile_jybeam": np.asarray(observed_profile),
        "model_profile_jybeam": np.asarray(model_profile),
        "annulus_pixel_counts": np.asarray(counts),
        "residual_rms_jybeam": float(
            np.sqrt(np.mean(residual[aperture]**2))
        ),
        "aperture_radius_mas": float(aperture_radius_mas),
        "aperture_radius_pixels": float(aperture_radius_mas / pixel_scale_mas),
        "aperture_selection": "observed_snr" if automatic_aperture else "explicit_radius",
    }

    if save_path is not None:
        fig, axes = plt.subplots(
            1, 4, figsize=(20, 4), constrained_layout=True
        )

        # Array coordinates relative to the chosen centre.
        # For the supplied orientation, east is towards decreasing columns.
        extent = [
            (-0.5 - xc) * pixel_scale_mas,
            (nx - 0.5 - xc) * pixel_scale_mas,
            (-0.5 - yc) * pixel_scale_mas,
            (ny - 0.5 - yc) * pixel_scale_mas,
        ]

        values = np.concatenate([observed[aperture], model[aperture]])
        vmin = min(0.0, float(values.min()))
        vmax = max(0.0, float(values.max()))
        if vmax <= vmin:
            vmax = vmin + 1.0

        for ax, image, title in zip(
            axes[:2],
            [observed, model],
            ["Observed", "Model"],
        ):
            im = ax.imshow(
                image,
                origin="lower",
                extent=extent,
                cmap="inferno",
                vmin=vmin,
                vmax=vmax,
            )
            fig.colorbar(im, ax=ax, label="Jy/beam")
            ax.set_title(title)

        residual_display = residual
        residual_label = "Observed − model [Jy/beam]"

        if noise_rms_jybeam is not None:
            noise = float(noise_rms_jybeam)
            if not np.isfinite(noise) or noise <= 0:
                plt.close(fig)
                raise ValueError("Noise RMS must be positive and finite.")
            residual_display = residual / noise
            residual_label = "(Observed − model) / background RMS"

        limit = float(np.max(np.abs(residual_display[aperture])))
        if limit == 0:
            limit = 1.0

        im = axes[2].imshow(
            residual_display,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        fig.colorbar(im, ax=axes[2], label=residual_label)
        axes[2].set_title("Signed residual")

        for ax in axes[:3]:
            ax.set_xlim(-aperture_radius_mas, aperture_radius_mas)
            ax.set_ylim(-aperture_radius_mas, aperture_radius_mas)
            ax.set_xlabel("Column offset [mas; west-positive]")
            ax.set_ylabel("Row offset [mas; north-positive]")
            ax.add_patch(plt.Circle(
                (0, 0), aperture_radius_mas,
                fill=False, color="cyan", linestyle="--",
            ))

        axes[3].plot(
            result["radius_mas"],
            result["observed_profile_jybeam"],
            "o-", label="Observed",
        )
        axes[3].plot(
            result["radius_mas"],
            result["model_profile_jybeam"],
            "o-", label="Model",
        )
        axes[3].set_xlabel("Radius [mas]")
        axes[3].set_ylabel("Annular mean [Jy/beam]")
        axes[3].legend()
        axes[3].set_title("Absolute radial brightness")

        fig.suptitle(
            f"Aperture radius: {aperture_radius_mas:g} mas | "
            f"Observed flux: {observed_flux:.5g} Jy | "
            f"Model flux: {model_flux:.5g} Jy | "
            f"Residual-energy loss: {residual_energy_loss:.4g}"
        )

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    return result
