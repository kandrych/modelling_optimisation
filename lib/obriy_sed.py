import distroi
import numpy as np
import pandas as pd
from typing import Literal, Tuple, Dict, Optional, Union, Any, List

from distroi.auxiliary import constants
from distroi.data import image
from distroi.data import sed
from distroi.model.geom_comp import geom_comp
from distroi.auxiliary import select_data_oifits
from distroi.data.oi_container import OIContainer

import os
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.units as u
from IPython.display import display
import glob
import subprocess
from skimage.transform import rescale, resize, downscale_local_mean
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft, AiryDisk2DKernel
import fnmatch
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import random


import astropy
from astropy import units as u
import astropy.units.quantity
from astropy.io import fits
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from textwrap import wrap
import scipy.ndimage as ndimage
from matplotlib.gridspec import GridSpec
from matplotlib import colors


from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from functools import wraps
import cv2

import shutil

from pathlib import Path

import lib.obriy_general as obg
import lib.obriy_mcfost as obm
import lib.obriy_polarimetry as obp
import lib.obriy_interferometry as obi




constants.set_matplotlib_params()  # set project matplotlib parameters




######################################
# SED
#######################################



def load_sed_data(data_filename:str):
    """
    Load SED data from a file and convert to appropriate units.

    Parameters
    ----------
    data_filename : str
        Path to the data file.
    Returns
    -------
    data_wave : np.ndarray
        Wavelengths in micrometers.
    data_flux : np.ndarray
        Fluxes in erg/s/cm^2.
    data_err : np.ndarray
        Flux errors in erg/s/cm^2.
    """
    #read in the data file with pandas
    df = pd.read_csv(data_filename, sep='\\s+', header=3, \
                            names=['meas', 'e_meas', 'flag', 'unit', 'photband', 'source',\
                                '_r', '_RAJ2000', '_DEJ2000', 'cwave', 'cmeas', 'e_cmeas',\
                                'cunit', 'color', 'include', 'phase', 'bibcode', 'comments'])
    #filter out the NaN values in certain column
    df = df[df['cwave'].notna()]
    data_wave = np.array(df['cwave'])*u.Angstrom
    data_flux = np.array(df['cmeas'])*(u.erg/u.second/(u.centimeter**2)/u.Angstrom)
    data_err = np.array(df['e_cmeas'])*(u.erg/u.second/(u.centimeter**2)/u.Angstrom)
    #put flux to lam x F_lam and in units erg cm^-2 s^-1
    data_wave = data_wave.to(u.micrometer)
    data_flux = (data_wave*data_flux).to(u.erg/u.second/(u.centimeter**2))
    data_err = (data_wave*data_err).to(u.erg/u.second/(u.centimeter**2))
    #extract only the values because this might cause trouble in sutractions later on
    if type(data_wave) == astropy.units.quantity.Quantity:
        data_wave = np.array(data_wave.value)
        data_flux = np.array(data_flux.value)
        data_err = np.array(data_err.value)
    return data_wave, data_flux, data_err



#define functions for fitting E(B-V) 
#'lam'=array of wavelengths 'flux'=flux values 'E'=E(B-V) magnitude 'path'=path to reddening law file
def redden_flux(lam, flux, path, E):
    """
    Applies ISM reddening to flux values using a specified reddening law.

    Parameters
    ----------
    lam : np.ndarray
        Wavelengths in micrometers.
    flux : np.ndarray
        Flux values in erg/s/cm^2.
    path : str
        Path to the reddening law file.
    E : float
        E(B-V) magnitude for reddening.
    Returns
    -------
    np.ndarray
        The reddened flux values in erg/s/cm^2.
    """
    if E == 0:
        return flux
    else:
        #read in the ISM reddening law wavelengths in Angström and A/E in magnitude
        df_law = pd.read_csv(path, header=2, names=['WAVE', 'A/E']\
                                , sep='\\s+', engine='python')
        #set wavelength to micrometer
        lam_law = np.array(df_law['WAVE'])*10**-4
        ae_law = np.array(df_law['A/E'])
        #creates a function to linearly interpolate A/E(B-V) values to used wavelengths
        f = interp1d(lam_law, ae_law, kind='linear', bounds_error=False, fill_value=0)
        ae = f(lam)
        return flux*10**(-ae*E/2.5)
    

def chi2reddened(lam, flux, lam_model, flux_model, flux_error, path, E, sigma_sys_frac = None):
    """
    Calculate reduced chi2 between data and model SED, with reddening applied to the model.
    Parameters
    ----------
    lam : np.ndarray
        Wavelengths of the data in micrometers.
    flux : np.ndarray
        Fluxes of the data in erg/s/cm^2.
    lam_model : np.ndarray
        Wavelengths of the model in micrometers.
    flux_model : np.ndarray
        Fluxes of the model in erg/s/cm^2.
    flux_error : np.ndarray
        Flux errors of the data in erg/s/cm^2.
    path : str
        Path to the reddening law file.
    E : float
        E(B-V) magnitude for reddening.
    Returns
    -------
    float
        The reduced chi2 value between the data and the reddened model. 
    float
        The full chi2 value between the data and the reddened model.
    float
        The log-likelihood value between the data and the reddened model.
    """
    #redden the model SEDx
    flux_model_red = redden_flux(lam_model, flux_model, path,  E)
    #interpolate the model to the wavelength values of the data
    f = interp1d(lam_model, flux_model_red)
    flux_model_red_interpol = f(lam)

    # Residuals
    resid = flux - flux_model_red_interpol

    # Per-point uncertainties (optionally include a systematic fractional term of model value)
    if sigma_sys_frac is not None:
        varience = flux_error**2 + (sigma_sys_frac * flux_model_red_interpol)**2
    else:
        varience = flux_error**2

    chi2_full = np.sum((resid**2) / varience)
    #calculate chi2
    chi2_reduced = chi2_full/(resid.shape[0]-1)

    # Gaussian log-likelihood (independent, heteroscedastic errors)
    loglike = -0.5 * np.sum((resid**2) / varience + np.log(2.0 * np.pi * varience))


    return chi2_reduced, chi2_full, loglike




def chi2_SED_with_reddening(
        folder_sim: str, 
        main_dir: str, 
        data_wave: np.ndarray,
        data_flux: np.ndarray,
        data_err: np.ndarray,
        reddening_law_path: str | None = None,
        plot: bool = True,
        description: str = None
        ) -> Tuple[float,float, float]:
    """
    Wrapper that loads mcfost SED data, calculate the reduced chi2 between the observed SED data 
    and the MCFOST model SED, including ISM reddening as a free parameter.
    Plots SED with and without reddening.
    Parameters
    ----------
    folder_sim : str
        The folder name where the MCFOST simulation is located.
    main_dir : str
        The main directory path where the simulation folder is located.
    data_wave : np.ndarray
        Wavelengths of the observed SED data in micrometers.
    data_flux : np.ndarray  
        Fluxes of the observed SED data in erg/s/cm^2.
    data_err : np.ndarray
        Flux errors of the observed SED data in erg/s/cm^2.
    reddening_law_path
        location anf file for the reddening
    plot : bool, optional
        If True, generates and saves plots of the SED with and without reddening. Default is True.
    description : str, optional
        An optional description to include in the plot title. Default is None.
    Returns
    -------
    float
        chi2 value between the observed SED data and the reddened MCFOST model SED.
    float
        The reduced chi2 
    float
        Log-likelihood 
    """

   # --- Resolve reddening law path if not provided ---
    if reddening_law_path is None:
        # folder_of_script = .../lib  (for example)
        folder_of_script = Path(__file__).resolve().parent
        # one above the folder of the script:
        reddening_law_path = str(folder_of_script.parent / "utils"/"ISMreddening_law_Cardelli1989.dat")

    # Optional: fail early with a clear message
    rpath = Path(reddening_law_path)
    if not rpath.is_file():
        raise FileNotFoundError(
            f"Reddening law not found at: {rpath}\n"
        )

    simulation_dir = main_dir+folder_sim+"/"
    #### read in data
    #open the required ray-traced SED fits file
    hdul=fits.open(simulation_dir+'data_th/sed_rt.fits.gz')
    #read in entire sed array and corresponding wavelength array
    sed_array=hdul[0].data*10**3 #converted from W/m**(2) as provided by MCFOST to cgs (erg/s/cm**2)
    lam=hdul[1].data
    #print('SED array:', sed_array)

    
    #single out full sed lambda times flux values
    #this is only so that the non-infinite minimum
    #can be easily found for setting the y-axis limits
    full_sed = sed_array[0, 0, 0, :]
    #single out the star only
    star_sed = sed_array[1, 0, 0, :]

    #open up the observed photometric data, note that the units are in erg/s/cm2/AA
    #cut out the laste points, these often overlap with the next column
    

    #### plot
    #plotting
    if plot:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(data_wave, data_flux, data_err, label='data', fmt='bd', mfc='white', capsize=5, zorder=1000)
        ax.plot(lam, full_sed, ls='-', c='k', label='MCFOST SED', zorder=1)
        ax.plot(lam, star_sed, ls='-', c='grey', label='star', alpha=0.4, zorder=0)
        ax.set_xlabel(r"$\lambda \, \mathrm{[\mu m]}$")
        ax.set_ylabel(r"$\lambda F_{\lambda} \, \mathrm{[erg \, cm^{-2} \, s^{-1}]}$")
        ax.set_xlim(np.min(lam), np.max(lam))
        ax.set_ylim(full_sed[np.isfinite(full_sed)].min(), full_sed[np.isfinite(full_sed)].max()*10)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('SED no ISM reddening, ' + folder_sim)
        ax.legend()
        plt.tight_layout()
        plt.savefig(simulation_dir+'sed_no_reddening.png', bbox_inches='tight')
        plt.close(fig)
    #plt.show()


    #### fit ISM reddening E(B-V)

    #run bootstrapping simulation assuming gaussian errorbars on SED data
    sim_number = 1
    E_values = np.zeros(sim_number)
    chi2_values_sim = np.zeros(sim_number)

    objective = obg.pick_output(chi2reddened, idx=1, cast=float)

    for i in range (0, sim_number):
        new_flux = np.zeros(data_wave.size)
        for j in range(0, data_wave.size):
            new_flux[j] = data_wave[j]+(random.gauss(0, 1)*data_err[j])
        par_min = minimize(lambda x: objective(data_wave, data_flux, lam, full_sed, data_err,\
                                                                reddening_law_path, x), 1.4)
        E_values[i] = par_min["x"][0]
        _,chi2_values_sim[i],_ = chi2reddened(data_wave, data_flux, lam, full_sed, data_err,\
                                        reddening_law_path, par_min["x"][0])

    E_best = np.mean(E_values)
    E_std = np.mean(np.std(E_values))
    #print('Mean value of E(B-V) is: ' + str(E_best))
    #print('Standard deviation of E(B-V) is:'  + str(E_std))

    #redden the model by found ISM reddenning and plot
    full_sed_red = redden_flux(lam, full_sed, reddening_law_path, E_best)
    star_sed_red = redden_flux(lam, star_sed, reddening_law_path, E_best)

    #calculate chi2 values
    chi2_red,chi2, loglike=chi2reddened(data_wave, data_flux, lam, full_sed, data_err, reddening_law_path, E_best)
    #plotting
    if plot:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.errorbar(data_wave, data_flux, data_err, label='data', fmt='bd', mfc='white', capsize=5, zorder=1000)
        ax.plot(lam, full_sed_red, ls='-', c='r', label='MCFOST SED reddened', zorder=1)
        ax.plot(lam, full_sed, ls='--', c='r', label='MCFOST SED no reddening', zorder=0, alpha=0.4)
        ax.plot(lam, star_sed_red, ls='-', c='k', label='STAR reddened', zorder=0)
        ax.plot(lam, star_sed, ls='--', c='k', label='STAR no reddening', alpha=0.4, zorder=0)
        ax.set_xlabel(r"$\lambda \, \mathrm{[\mu m]}$")
        ax.set_ylabel(r"$\lambda F_{\lambda} \, \mathrm{[erg \, cm^{-2} \, s^{-1}]}$")
        ax.set_xlim(np.min(lam), np.max(lam))
        ax.set_ylim(full_sed[np.isfinite(full_sed)].min(), full_sed[np.isfinite(full_sed)].max()*10)
        ax.set_xscale('log')
        ax.set_yscale('log')
        fig.suptitle("SED, data and RT", fontsize=16, y=0.96)

        desc = description if description is not None else ""
        ax.text(0.5, 1.08, desc,
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=10, wrap=True)  

        ax.text(0.5, 1.03,
            r"$\chi^2 reduced = $" + f"{chi2_red:.2f}",
            ha='center', va='bottom',
            transform=ax.transAxes,
            fontsize=12)
        ax.legend()
        plt.tight_layout()

        fig.savefig(simulation_dir+'SED_Akke_model1_MCFOST.png', dpi= 300, bbox_inches='tight')
        plt.close(fig)
        
    
    return  chi2, chi2_red, loglike

