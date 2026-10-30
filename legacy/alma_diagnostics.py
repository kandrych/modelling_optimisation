"""Historical ALMA resampling, cropping and profile-score diagnostics.

These functions are retained for old analyses and are not used by SMAC3.
Their numerical definitions are unchanged. chi2_ALMA is not a replacement
for the supported image comparison in lib.obriy_alma.compare_alma_images.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
from skimage.transform import rescale


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
    # These imports are needed only for this historical profile diagnostic.
    from lib import obriy_general as obg
    from lib import obriy_polarimetry as obp

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
        with obg.diagnostic_plot("ALMA model intensity"):
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

