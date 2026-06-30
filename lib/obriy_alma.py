import os
import fnmatch
from astropy.io import fits
from pathlib import Path
import numpy as np
from typing import Literal, Tuple, Dict, Optional, Union, Any, List
import subprocess

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
    img_sim: np.ndarray,
    img_data: np.ndarray
) -> Tuple[np.ndarray]:
    """
    Cut down the ALMA image to match the size of the data image.

    Parameters
    ----------
    img_sim : np.ndarray
        Simulated image.
    img_data : np.ndarray
        Observed data image.
    
    Returns
    -------
    img_tot_res
    """
    size_sim = img_sim.shape[0]
    size_data = img_data.shape[0]
    if size_sim <= 0 or size_data <= 0:
        raise ValueError("Image dimensions must be positive.")
    
    if size_sim < size_data:
        raise ValueError("Simulated image must be larger than data image. Size of simulation: {}, size of data: {}".format(size_sim, size_data))
    
    diff = (size_sim - size_data) // 2
    if (size_sim - size_data) % 2 != 0:
        raise ValueError("Simulated image size must be even larger than data image size. Size of simulation: {}, size of data: {}".format(size_sim, size_data))

    img_tot_res = img_sim[diff:size_sim-diff, diff:size_sim-diff]
   

    return img_tot_res

def chi2_ALMA(main_dir, data_alma, plot=False, fig_dir=None, extra_title=""):
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
    # Load the simulation data for ALMA
    _, _, simulated_itot, _, _, _, _, _, _, _ = obp.load_mcfost_images_1wave(str(main_dir), '870.0')  
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