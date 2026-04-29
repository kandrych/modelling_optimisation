import lib.obriy_general as obg
import lib.obriy_interferometry as obi
import lib.obriy_sed as obs
import lib.obriy_mcfost as obm
import lib.obriy_polarimetry as obp



def chi2_ALMA(main_dir, data_alma, plot=False, description=""):
    """
    Compute the chi2 for ALMA data.
    """
    alma_cont = data_alma['alma_cont']
    ps_alma = data_alma['ps_alma']
    alma_wavelength = data_alma['alma_wavelength']
    # Load the simulation data for ALMA
    simulated_array, simulated_header, simulated_itot, _, _, _, _, _, _, _ = obp.load_mcfost_images_1wave(main_dir, '870.0')  
    #compute profiles
    prof_rad = radial_br_profile(
            model_data, ps,
            inclination_deg=inc_deg,
            position_angle_deg=pa_deg,
            R_limit=radial_limit_mas,          # <-- if it expects pixels
            mode=mode, xc=xc, yc=yc,
            plot=plot,
            save=save_prefix + "radial_"
        )


    return chi2_alma, chi2_red_alma, loglike_alma