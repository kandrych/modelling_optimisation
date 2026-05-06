import os
import fnmatch
from astropy.io import fits
from pathlib import Path

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

            
    return data, header

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
                                                radial_limit_mas=500,
                                                plot=plot,
                                                save_prefix=str(fig_dir)+extra_title+'_profile_',
                                                deprojection_inc_pa_deg=(0.0, 0.0),
                                                center=None,
                                                az_nbins=18,
                                                azimuthal_r_in_mas=0.0,
                                                azimuthal_r_out_mas=500.0,
                                                theta0=0.0
                                                )
    
    profile_rad_pi_chi2, _,_, profile_rad_pi_npoints = obp.profile_chi2(obs_rad_prof, radial_profile_alma_model, ps_alma, profile_type="radial", plot=plot, save_prefix=str(fig_dir)+extra_title+'_radial_profile_')
    profile_az_pi_chi2, _,_, profile_az_pi_npoints = obp.profile_chi2(obs_az_prof, azimuthal_profile_alma_model, ps_alma, profile_type="azimuthal", plot=plot, save_prefix=str(fig_dir)+extra_title+'_azimuthal_profile_')
    profiles_chi2_red= (profile_rad_pi_chi2 + profile_az_pi_chi2) / (profile_rad_pi_npoints + profile_az_pi_npoints -2)
            


    return profiles_chi2_red, profile_rad_pi_chi2, profile_az_pi_chi2, profile_rad_pi_npoints, profile_az_pi_npoints