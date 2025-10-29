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
import pymcfost as mcfost

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





constants.set_matplotlib_params()  # set project matplotlib parameters

# Ensure MCFOST is found in PATH on Katya's Mac
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]
os.environ["MCFOST_UTILS"] = os.path.expanduser("/Users/katerynaandrych/software/mcfost/utils")




###################################
# GENERAL
###################################


def pick_output(fn, idx=0, cast=float):
    """
    Calls fn(*args, **kwargs), returns the element at idx.
    Optionally casts to a scalar type (default: float).
    """
    @wraps(fn)
    def _wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        try:
            val = out[idx]
        except TypeError:
            raise TypeError(f"{fn.__name__} did not return an indexable object, got {type(out)}")
        except IndexError:
            raise IndexError(f"Index {idx} out of range for output {out!r}")
        return cast(val) if cast is not None else val
    return _wrapped


def mas2au(angle_mas: float, distance_pc: float) -> float:
    """
    Convert angular size in milliarcseconds (mas) to linear size in astronomical units (au).

    Parameters
    ----------
    angle_mas : float
        Angular size in milliarcseconds (mas).
    distance_pc : float
        Distance to the object in parsecs (pc).

    Returns
    -------
    size_au : float
        Linear size in astronomical units (au).
    """
    # 1 mas = 1e-3 arcsec, and 1 arcsec at 1 pc = 1 au
    size_au = angle_mas * 1e-3 * distance_pc
    
    #analogous calculation using astropy units
    #size_au = (angle_mas*u.mas).to(u.rad)*(distance_pc)*u.parsec.to(u.au) 

    return size_au



def write_header(path: str) -> None:
    """
    Write the header line to the output file.
    Parameters
    ----------
    path : str
        Path to the output file.
    Returns
    -------
    None
    """
    with open(path, "w") as f:
        f.write(
            f"{'instrument':<10} {'band':<5} {'mode':<13} {'background':<10} {'vistype':<8} "
            f"{'wave_min':<10} {'wave_max':<10} {'chi2':<12} {'chi2_red':<12} {'num_points':<12}\n"
        )

def record(
    path: str,
    instrument: str,
    band: str,
    mode: str,
    background: float,
    vistype: str,
    chi2: float,
    chi2_red: float,
    num_points: int = 0,
    wave_lims: tuple[float, float] | None = None
) -> None:
    """
    Record the fit results (given as parameters) to a file.

    Parameters
    ----------
    path : str
        Path to the output file.
    instrument : str
        Instrument name.
    band : str
        Observing band.
    mode : str
        Observing mode.
    background : float
        Background flux fraction.
    vistype : {'vis2', 'vis', 'fcorr'}
        Type of visibility.
    chi2 : float
        Chi2 value.
    chi2_red : float
        Reduced chi2 value.
    num_points : int, optional
        Number of data points used in the fit. Default is 0.
    wave_lims : tuple of float, optional
        Wavelength limits (min, max) in micrometers. Default is None.
    Returns
    -------
    None
    """
    wmin, wmax = ("", "") if wave_lims is None else (f"{wave_lims[0]:.5g}", f"{wave_lims[1]:.5g}")
    with open(path, "a") as f:
        f.write(
            f"{instrument:<10} {band:<5} {mode:<13} {background:<10} {vistype:<8} "
            f"{wmin:<10} {wmax:<10} {chi2:<12.6g} {chi2_red:<12.6g} {num_points:<10}\n"
        )




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
        reddening_law_path: str = '/Users/katerynaandrych/Work/lin/Postdoc/Data/SED_reddening/ISMreddening_law_Cardelli1989.dat',
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

    objective = pick_output(chi2reddened, idx=1, cast=float)

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
    chi2, chi2_red, loglike=chi2reddened(data_wave, data_flux, lam, full_sed, data_err, reddening_law_path, E_best)
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






#######################################
# INTERFEROMETRY
#######################################


def oi_container_chi2(
    container_data,
    container_model,
    vistype: Literal["vis2", "vis"] = "vis2",
    sigma_sys_frac: float=None
) -> Tuple[float, float, float, int]:
    """
    Calculate the reduced chi2 between data and model contained in two OIContainer objects.

    Parameters
    ----------
    container_data : distroi.data.oi_container.OIContainer
        OIContainer object containing the observational data.
    container_model : distroi.data.oi_container.OIContainer
        OIContainer object containing the model observables.
    vistype : {'vis2', 'vis', 'fcorr'}, optional
        Type of visibility to use for chi2 calculation (default is 'vis2').
    
    Returns
    -------
    chi2_red : float
        Reduced chi2 value.
     """
    if vistype not in ['vis2', 'vis']:
        raise ValueError("vistype must be one of 'vis2', 'vis'.")
    
    # Initialize chi2 accumulators
    chi2_sum = 0.0
    loglike_sum=0.0
    n_data_points = 0

    # set spatial frequencies, visibilities and plotting label based on specified option
    if vistype == "vis2":
        vismod = container_model.v2
        visdata = container_data.v2
        viserrdata = container_data.v2_err
        wavedata = container_data.v2_wave
        basedata = container_data.v2_base
        
    elif vistype == "vis":
        vismod = container_model.v
        wavedata = container_data.v_wave
        visdata = container_data.v
        viserrdata = container_data.v_err
        basedata = container_data.v_base
        if (not container_data.vis_in_fcorr) and (not container_model.vis_in_fcorr):
            vislabel = "$V$"
        elif container_data.vis_in_fcorr and container_model.vis_in_fcorr:
            vislabel = r"$F_{corr}$ (Jy)"
        else:
            raise Exception("container_data and container_mod do not have the same value for vis_in_fcorr")

    
    if visdata.shape != vismod.shape or visdata.shape != viserrdata.shape:
        raise ValueError(
            f"Shape mismatch: data{visdata.shape}, model.v2{vismod.shape}, data.v2_err{viserrdata.shape}"
        )
   
    # Loop over all data points
    for i in range(len(visdata)):
        if viserrdata[i] > 0:
            if sigma_sys_frac is not None:
                varience = viserrdata[i] ** 2 + (sigma_sys_frac * vismod[i])**2
            else:
                varience = viserrdata[i] ** 2
            chi2_sum += ((visdata[i] - vismod[i]) ** 2) / varience
            loglike_sum+=((visdata[i] - vismod[i]) ** 2)/varience +np.log(2.0 * np.pi * varience)
            n_data_points += 1  # Count only points with valid error bars

    if n_data_points == 0:
        raise ValueError("No valid data points with positive error bars found for chi2 calculation.")
    chi2_red = chi2_sum / (n_data_points-1)
    loglike=-0.5*loglike_sum
    
    return chi2_sum, chi2_red, loglike, n_data_points



def oi_container_plot_data_vs_model(
    container_data: OIContainer,
    container_mod: OIContainer,
    fig_dir: str = None,
    log_plotv: bool = False,
    plot_vistype: Literal["vis2", "vis", "fcorr"] = "vis2",
    show_plots: bool = True,
    chi_plot: str = None,
    extra_title: str = None
) -> None:
    """
    Plots the data against the model OI observables. Currently, plots uv coverage, a (squared) visibility curve and
    closure phases. Note that this function shares a name with a similar function in the sed module. Take care with
    your namespace if you use both functions in the same script.

    Parameters
    ----------
    container_data : OIContainer
        Container with data observables.
    container_mod : OIContainer
        Container with model observables.
    fig_dir : str, optional
        Directory to store plots in.
    log_plotv : bool, optional
        Set to True for a logarithmic y-scale in the (squared) visibility plot.
    plot_vistype : {'vis2', 'vis', 'fcorr'}, optional
        Sets the type of visibility to be plotted. 'vis2' for squared visibilities, 'vis' for visibilities or 'fcorr'
        for correlated flux in Jy.
    show_plots : bool, optional
        Set to False if you do not want the plots to be shown during your python instance. Note that if True, this
        freezes further code execution until the plot windows are closed.
    chi_plot : float, optional
        If provided, the reduced chi2 value will be indicated in the (squared) visibility plot.
    extra_title : str, optional
        Additional string to append to the figure title and saved filename.

    Returns
    -------
    None
    """
    valid_vistypes = ["vis2", "vis", "fcorr"]
    if plot_vistype not in valid_vistypes:
        raise ValueError(f"Warning: Invalid plot_vistype '{plot_vistype}'. Valid options are: {valid_vistypes}.")
    
    # create plotting directory if it doesn't exist yet
    if fig_dir is not None:
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)

    # set spatial frequencies, visibilities and plotting label based on specified option
    if plot_vistype == "vis2":
        ufdata = container_data.v2_uf
        vfdata = container_data.v2_vf
        vismod = container_mod.v2
        visdata = container_data.v2
        viserrdata = container_data.v2_err
        wavedata = container_data.v2_wave
        basedata = container_data.v2_base
        vislabel = "$V^2$"
    elif plot_vistype == "vis" or plot_vistype == "fcorr":
        ufdata = container_data.v_uf
        vfdata = container_data.v_vf
        vismod = container_mod.v
        wavedata = container_data.v_wave
        visdata = container_data.v
        viserrdata = container_data.v_err
        basedata = container_data.v_base
        if (not container_data.vis_in_fcorr) and (not container_mod.vis_in_fcorr):
            vislabel = "$V$"
        elif container_data.vis_in_fcorr and container_mod.vis_in_fcorr:
            vislabel = r"$F_{corr}$ (Jy)"
        else:
            raise Exception("container_data and container_mod do not have the same value for vis_in_fcorr")
            return
    
    # plot uv coverage
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
    ax.set_aspect("equal", adjustable="datalim")  # make plot axes have the same scale
    ax.scatter(
        ufdata / 1e6,
        vfdata / 1e6,
        c=wavedata,
        s=1,
        cmap=constants.PLOT_CMAP,
    )
    sc = ax.scatter(
        -ufdata / 1e6,
        -vfdata / 1e6,
        c=wavedata,
        s=1,
        cmap=constants.PLOT_CMAP,
    )
    clb = fig.colorbar(sc, cax=cax)
    clb.set_label(r"$\lambda$ ($\mu$m)", labelpad=5)

    ax.set_xlim(ax.get_xlim()[::-1])  # switch x-axis direction
    ax.set_title("uv coverage")
    ax.set_xlabel(r"$\leftarrow B_u$ ($\mathrm{M \lambda}$)")
    ax.set_ylabel(r"$B_v \rightarrow$ ($\mathrm{M \lambda}$)")
    
    
    if fig_dir is not None:
        plt.savefig(
            os.path.join(fig_dir, f"uv_plane.{constants.FIG_OUTPUT_TYPE}"),
            dpi=constants.FIG_DPI,
            bbox_inches="tight",
        )
    plt.close(fig)

    # plot (squared) visibilities
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[1, 0.3])
    ax = gs.subplots(sharex=True)

    ax[0].errorbar(
        basedata,
        visdata,
        viserrdata,
        label="data",
        mec="royalblue",
        marker="o",
        capsize=0,
        zorder=0,
        markersize=2,
        ls="",
        alpha=0.8,
        elinewidth=0.5,
    )
    ax[0].scatter(
        basedata,
        vismod,
        label="model",
        marker="o",
        facecolor="white",
        edgecolor="r",
        s=4,
        alpha=0.6,
    )
    ax[1].scatter(
        basedata,
        (vismod - visdata) / viserrdata,
        marker="o",
        facecolor="white",
        edgecolor="r",
        s=4,
        alpha=0.6,
    )

    ax[0].set_ylabel(vislabel)
    ax[0].legend()
    ax[0].set_title(f"Visibilities{f' {extra_title}' if extra_title else ''}", pad=30)  # increase pad (default is ~6)

    # Place text just below the title in axes coordinates
    ax[0].text(
        0.5, 1.02,   # centered horizontally, slightly above the top of the axes
        r"$\chi_{red}^2 = $" + f"{chi_plot:.2f}",
        ha='center', va='bottom',
        fontsize=12,
        transform=ax[0].transAxes
    )
    ax[0].tick_params(axis="x", direction="in", pad=-15)

    if log_plotv:
        ax[0].set_ylim(0.5 * np.min(visdata), 1.1 * np.max(np.maximum(visdata, vismod)))
        ax[0].set_yscale("log")
    else:
        ax[0].set_ylim(0, 1.1 * np.max(np.maximum(visdata, vismod)))

    ax[1].set_xlim(0, np.max(basedata) * 1.05)
    ax[1].axhline(y=0, c="k", ls="--", lw=1, zorder=0)
    ax[1].set_xlabel(r"$B$ ($\mathrm{M \lambda}$)")
    ax[1].set_ylabel(r"error $(\sigma)$")
    if fig_dir is not None:
        plt.savefig(
            os.path.join(fig_dir, f"visibilities{f'_{extra_title}' if extra_title else ''}.{constants.FIG_OUTPUT_TYPE}"),
            dpi=constants.FIG_DPI,
            bbox_inches="tight",
        )
    if show_plots:
        plt.show(fig)
    plt.close(fig)
    # plot phi_closure
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[1, 0.3])
    ax = gs.subplots(sharex=True)

    ax[0].errorbar(
        container_data.t3_bmax,
        container_data.t3_phi,
        container_data.t3_phierr,
        label="data",
        mec="royalblue",
        marker="o",
        capsize=0,
        zorder=0,
        markersize=2,
        ls="",
        alpha=0.8,
        elinewidth=0.5,
    )
    ax[0].scatter(
        container_data.t3_bmax,
        container_mod.t3_phi,
        label="model",
        marker="o",
        facecolor="white",
        edgecolor="r",
        s=4,
        alpha=0.6,
    )
    ax[1].scatter(
        container_data.t3_bmax,
        (container_mod.t3_phi - container_data.t3_phi) / container_data.t3_phierr,
        marker="o",
        facecolor="white",
        edgecolor="r",
        s=4,
        alpha=0.6,
    )

    ax[0].set_ylabel(r"$\phi_{CP}$ ($^\circ$)")
    ax[0].legend()
    ax[0].set_title(f"Closure Phases{f' {extra_title}' if extra_title else ''}")
    ax[0].tick_params(axis="x", direction="in", pad=-15)
    ax[0].set_ylim(
        min(
            np.min(container_data.t3_phi - container_data.t3_phierr),
            np.min(container_mod.t3_phi),
        ),
        max(
            np.max(container_data.t3_phi + container_data.t3_phierr),
            np.max(container_mod.t3_phi),
        ),
    )

    ax[1].set_xlim(0, np.max(container_data.t3_bmax) * 1.05)
    ax[1].axhline(y=0, c="k", ls="--", lw=1, zorder=0)
    ax[1].set_xlabel(r"$B_{max}$ ($\mathrm{M \lambda}$)")
    ax[1].set_ylabel(r"error $(\sigma_{\phi_{CP}})$")
    if fig_dir is not None:
        plt.savefig(
            os.path.join(fig_dir, f"closure_phases{f'_{extra_title}' if extra_title else ''}.{constants.FIG_OUTPUT_TYPE}"),
            dpi=constants.FIG_DPI,
            bbox_inches="tight",
        )
    if show_plots:
        plt.show(fig)
    plt.close(fig)

    return





def chi2_for_optimisation_overresolved(frac, ref_wavelength, container_data, img_ffts) -> Tuple[float, float, float, int]:
    """
    Adding the background component to the model and calculating the reduced chi2 between the observed interferometric data and the model with an overresolved background component.
    
    Parameters
    ----------
    frac : float
        The flux fraction of the overresolved background component.
    ref_wavelength : float      
        The reference wavelength in micrometers.
    container_data : OIContainer
        The container with observed interferometric data.
    img_ffts : list of ImageFFT
        The list of mcfost model ImageFFT objects for different wavelengths.
    Returns
    -------
    float
        Chi2 value between the observed data and the model with the overresolved background component.
    float
        The reduced chi2 value.
    float
        Likelihood for optimisation
    Int 
        Number of data points
    """
    frac = float(frac) 
    background = distroi.Overresolved(sp_dep=distroi.FlatSpecDep(flux_form="flam"))
    container_model = distroi.oi_container_calc_image_fft_observables(
        container_data, img_ffts, geom_comps=[background], geom_comp_flux_fracs=[frac], ref_wavelength=ref_wavelength)


    chi2, chi2_red, loglike, n_data=oi_container_chi2(container_data, container_model)
    return chi2, chi2_red, loglike, n_data

def monochromatic_chi(
        simulation_dir: str,
        img_dir: str,
        container_data: OIContainer,
        vistype: str='vis2',
        plot: bool=False,
        fig_dir: str=None,
        extra_title: str=None,
        log_plotv: bool=False
) -> Tuple[float, float,float, int]:
    """
    Wrapper to calculate chi2 and reduced chi2 for a monochromatic model without background.

    ----------
    Parameters
    
    simulation_dir : str
        Directory where the MCFOST simulation is located.
    img_dir : str
        Directory where the MCFOST image for specific wavelength is located.
    container_data : OIContainer
        Container with data observables.
    vistype : {'vis2', 'vis', 'fcorr'}, optional
        Type of visibility to be used in the chi2 calculation. Default is 'vis2'.
    plot : bool, optional
        If True, plots data vs model. Default is False.
    fig_dir : str, optional
        Directory to save plots if plot is True. Default is None.
    extra_title : str, optional
        Extra title to add to the plots if plot is True. Default is None.
    log_plotv : bool, optional
        If True, plots visibility in logarithmic scale. Default is False.

    -------
    Returns
    
    chi2 : float
        Chi2 value.
    chi2_red : float
        Reduced chi2 value.
    likelihood : float
        Log-likelihood value for optimisation.
    num_points : int
        Number of data points used in the chi2 calculation.
    """
    
    img_ffts = distroi.read_image_list(simulation_dir, img_dir)
    container_model = distroi.oi_container_calc_image_fft_observables(container_data, img_ffts)
    chi2, chi2_red, likelihood, num_points=oi_container_chi2(container_data, container_model, vistype=vistype)

    if plot:    
        oi_container_plot_data_vs_model(
            container_data,
            container_model,
            fig_dir=fig_dir,
            log_plotv=log_plotv,
            plot_vistype=vistype,
            show_plots=False,
            chi_plot=chi2_red,
            extra_title=extra_title)

    return chi2, chi2_red, likelihood, num_points



def monochromatic_chi_with_background(
        simulation_dir: str,
        img_dir: str,
        container_data: OIContainer,
        wave_for_background: float,
        vistype: str='vis2',
        plot: bool=False,
        fig_dir: str=None,
        extra_title: str=None,
        log_plotv: bool=False
) -> Tuple[float, float, float, int]:
    """
    Calculate chi2 and reduced chi2 for a monochromatic model with background.
    Background is optimised for a given wavelength based on the reduced chi2.
    ----------
    Parameters

    simulation_dir : str
        Directory where the MCFOST simulation is located.
    img_dir : str
        Directory where the MCFOST image for specific wavelength is located.
    container_data : OIContainer
        Container with data observables.
    wave_for_background : float
        Wavelength in micrometer for which the background is calculated.
    vistype : {'vis2', 'vis', 'fcorr'}, optional
        Type of visibility to be used in the chi2 calculation. Default is 'vis2'.
    plot : bool, optional
        If True, plots data vs model. Default is False.
    fig_dir : str, optional
        Directory to save plots if plot is True. Default is None.
    extra_title : str, optional
        Extra title to add to the plots if plot is True. Default is None.
    log_plotv : bool, optional
        If True, plots visibility in logarithmic scale. Default is False.
    -------
    Returns
    chi2 : float
        Chi2 value.
    chi2_red : float
        Reduced chi2 value.
    loglike : float
        Log-likelihood value for optimisation.
    num_points : int
        Number of data points used in the chi2 calculation.
    """
    
    
    img_ffts = distroi.read_image_list(simulation_dir, img_dir)
    
    background = distroi.Overresolved(sp_dep=distroi.FlatSpecDep(flux_form="flam"))
    objective = pick_output(chi2_for_optimisation_overresolved, idx=2, cast=float)

    frac_min = minimize_scalar(
        lambda x: objective(x, ref_wavelength=wave_for_background),
        bounds=(0.0, 0.5),
        method="bounded",
        options={"xatol": 1e-4}
    )
    frac_best = float(frac_min.x)
    
    container_model = distroi.oi_container_calc_image_fft_observables(
        container_data, img_ffts, geom_comps=[background], geom_comp_flux_fracs=[frac_best], ref_wavelength=wave_for_background
    )

    chi2, chi2_red,loglike, num_points=oi_container_chi2(container_data, container_model, vistype=vistype)

    if plot:
        oi_container_plot_data_vs_model(
            container_data,
            container_model,
            fig_dir=fig_dir,
            log_plotv=log_plotv,
            plot_vistype=vistype,
            show_plots=False,
            chi_plot=chi2_red,
            extra_title=extra_title+f" with background fraction {frac_best:.3f}"
        )

    return chi2, chi2_red, loglike, num_points



###################################################################################
# POLARIMETRY
###################################################################################

IRDIS_PIX_MAS = 12.27
ZIMPOL_PIX_MAS = 3.6

def load_mcfost_images_1wave(
        main_dir: str,
        wavelength: str, 
        *,
        ploting: bool= False, 
        save_plots: str = None, 
        title_addition:str = ''
) -> tuple[np.ndarray, fits.Header, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and optionally visualize MCFOST model images at a single wavelength.

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
        img_q : np.ndarray
            Stokes Q image.
        img_u : np.ndarray
            Stokes U image.
        img_v : np.ndarray
            Stokes V image.
        img_star : np.ndarray
            Direct stellar light image.
        img_star_sct : np.ndarray
            Stellar scattered light image.
        img_disk_th : np.ndarray
            Disk thermal emission image.
        img_disk_th_sct : np.ndarray
            Scattered thermal emission image.

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
        #load in all images and separate 
        img_array=hdul[0].data
        #total intensity
        img_tot=img_array[:][:][0][0][0]
        #stokes Q intensity
        img_q=img_array[:][:][1][0][0]
        #stokes U intensity
        img_u=img_array[:][:][2][0][0]
        #stokes V intensity
        img_v=img_array[:][:][3][0][0]
        #direct starlight intensity
        img_star=img_array[:][:][4][0][0]
        #scattered starlight intensity
        img_star_sct=img_array[:][:][5][0][0]
        #disk thermal intensity
        img_disk_th=img_array[:][:][6][0][0]
        #disk scattered thermal intensity
        img_disk_th_sct=img_array[:][:][7][0][0]
        
        if ploting==True:
            #do some plotting
            fig, ax = plt.subplots(2, 4, figsize=(14,7))
            color_map = 'viridis' #'afmhot'
            ax[0][0].imshow(img_tot, color_map, extent=[+img_tot.shape[0]/2, -img_tot.shape[0]/2, -img_tot.shape[1]/2, img_tot.shape[1]/2])
            ax[0][0].set_title('$I_{tot}$')
            ax[0][1].imshow(img_q, color_map)
            ax[0][1].set_title('$Q$')
            ax[0][2].imshow(img_u, color_map)
            ax[0][2].set_title('$U$')
            ax[0][3].imshow(img_v, color_map)
            ax[0][3].set_title('$V$')
            ax[1][0].imshow(img_tot-img_star, color_map,extent=[+img_tot.shape[0]/2, -img_tot.shape[0]/2, -img_tot.shape[1]/2, img_tot.shape[1]/2])
            ax[1][0].set_title('$I_{disk}$')
            #ax[1][0].set_xlim([-img_tot.shape[0]/6, img_tot.shape[0]/6])
            #ax[1][0].set_ylim([-img_tot.shape[1]/6, img_tot.shape[1]/6])
            ax[1][1].imshow(img_disk_th, color_map)
            ax[1][1].set_title('$I_{disk,th}$')
            ax[1][2].imshow(img_star_sct, color_map)
            ax[1][2].set_title('$I_{disk,scat,*}$')
            ax[1][3].imshow(img_disk_th_sct, color_map)
            ax[1][3].set_title('$I_{disk,scat,th}$')
            plt.suptitle(str(wave)+r' $\mu m$, '+title_addition)
            plt.tight_layout()
            #plt.show()
            #save the plots
            if save_plots:
                fig.savefig(save_plots+title_addition+' image_decomp_'+str(wave)+'.png', dpi= 150, bbox_inches='tight')
            plt.close(fig)
        return img_array, header_data, img_tot, img_q, img_u, img_v, img_star, img_star_sct, img_disk_th, img_disk_th_sct




def synthetic_psf(ps, psf_FWHM):
    """
    Generate a synthetic 2D Gaussian point-spread function (PSF).

    Parameters
    ----------
    ps : float
        Pixel scale in milliarcseconds (mas/pixel).
    psf_FWHM : float
        Full width at half maximum (FWHM) of the PSF in milliarcseconds (mas).

    Returns
    -------
    psf : Gaussian2DKernel
        A 2D Gaussian kernel representing the synthetic PSF.

    Notes
    -----

    The kernel size is set to 15 * sigma in both x and y directions to ensure
    sufficient coverage of the Gaussian wings.
    """
    
    sigma = psf_FWHM / (ps * 2*np.sqrt(2*np.log(2)))
    psf = Gaussian2DKernel(sigma,x_size=int(15*sigma),y_size=int(15*sigma))
    return psf

def Loadimage(dirdat,filename):
    """
    Loading reduced data fits from SPHERE

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

            
    return fit



def gaus(x,a,x0,sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))


def find_FWHM (PSF,n):             #resolution
    middle=int(n/2)

    y1=PSF[middle,:]
    y2=PSF[:,middle]

    xdata = np.linspace(0,n, num=len(y1))


    n_gauss = len(xdata) #the number of data
    amp=np.max(y1)
    mean = np.sum(xdata * y1) / sum(y1)
    sigma = np.sqrt(sum(y1 * (xdata - mean)**2) / sum(y1))

    popt1,pcov1 = curve_fit(gaus,xdata,y1,p0=[amp,mean,sigma])
    popt2,pcov2 = curve_fit(gaus,xdata,y2,p0=[amp,mean,sigma])

    fwhm1=2*np.sqrt(2*math.log(2))*popt1[2]
    fwhm2=2.355*popt2[2]


    fwhm=(abs(fwhm1)+abs(fwhm2))/2

    return fwhm



def calculate_unresolved(correction_radius, q, u,i,ps,R,normlim):
    """
    Calculate the degree and angle of unresolved polarisation and correct Q and U images.   
    Parameters
    ----------
    correction_radius : float
        Radius in pixels within which to calculate unresolved polarisation.
    q : np.ndarray
        Stokes Q image.
    u : np.ndarray
        Stokes U image.
    i : np.ndarray
        Total intensity image.
    ps : float
        Pixel scale in mas/pixel.
    R : np.ndarray
        Radial distance array from image center.
    normlim : float
        Radius in pixels used for  the calculation of degree of unresolved polarisation. 
    Returns
        -------
        dolp_unres : float
            Degree of unresolved polarisation (fraction).
        aolp_unres : float
            Angle of unresolved polarisation (degrees).
        q_corr : np.ndarray
            Corrected Stokes Q image.
        u_corr : np.ndarray
            Corrected Stokes U image.
        """
    # Calculates degree and angle of unresolved polarisation
    #resulting values are in fraction (not %) for dolp, and in degrees for aolp
    

    mask=(R<=correction_radius)

    normalisation=np.sum(i[R<=1500/ps]) #normalisation within 1500 mas from central star
    q_over_i=np.divide(q,i,where=i!=0)   
    cq=np.median(q_over_i[mask]) #for median normal as in IRDIS
    u_over_i=np.divide(u,i,where=i!=0)    
    cu=np.median(u_over_i[mask]) #for median normal as in IRDIS
    aolp_unres=np.rad2deg(0.5*np.arctan2(cu, cq))
    #print(aolp_unres)
    if aolp_unres<0 : 
        aolp_unres=aolp_unres+180
    dolp_unres=np.sum(np.sqrt(cu*i*cu*i+ cq*i*cq*i)*(R<=normlim))/normalisation
    
    q_corr=q-cq*i
    u_corr=u-cu*i
    return dolp_unres, aolp_unres,q_corr,u_corr



def compute_grid(img: np.ndarray, 
                 xc:Optional[float]=None, 
                 yc:Optional[float]=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate centered coordinate grids for a square image.
    The grid accounts for half-pixel shifts in the case of even-sized arrays.

    Parameters
    ----------
    img : ndarray
        Input 2D square image. Only the first dimension (size) is used.

    Returns
    -------
    R : ndarray
        Radial distance array from the image center, shape (n, n).
    x : ndarray
        1D x-coordinates, length n.
    y : ndarray
        1D y-coordinates, length n.
    X : ndarray
        2D meshgrid of x, shape (n, n).
    Y : ndarray
        2D meshgrid of y, shape (n, n).

    Notes
    -----
    - If the image has even size `n`, the grid is shifted by -0.5 to ensure
      proper centering.
    - The coordinate system runs from negative to positive around the image center.
    """
    if img.ndim != 2:
        raise ValueError("img must be 2D")

    ny, nx = img.shape
    if xc is None:
        xc = (nx - 1) / 2.0   # exact center for odd/even sizes
    if yc is None:
        yc = (ny - 1) / 2.0

    x = np.arange(nx, dtype=float) - xc
    y = np.arange(ny, dtype=float) - yc
    X, Y = np.meshgrid(x, y)
    
    R = np.sqrt(X**2 + Y**2)
    return R, x, y, X, Y

def plot_polarimetric_image(
    image_to_plot: np.ndarray,
    ps_mas: Optional[float],
    *,
    image_scale: str = "linear",
    title: str = "",
    save: Optional[str] = None,
    show: bool = True,
    roi_half_size: Optional[int] = None,
    roi_center: Optional[Tuple[int, int]] = None,
    cmap_image: str = "viridis",
    cbar_label: Optional[str] = None,
    return_fig_ax: bool = False,
    aolp_quiver: bool = False,
    bin_factor: Tuple[int, int] = (4, 4),
    Q: Optional[np.ndarray]=None,
    U: Optional[np.ndarray]=None,
    I: Optional[np.ndarray]=None,
    snr_threshold: Optional[float] = 1,
    noise_level: Optional[float] = None,
    quiver_scale: Optional[float] = 0.1
) -> Optional[Tuple[plt.Figure, np.ndarray]]:
    """
    Status: not fully verified

    Plot an image with optional AoLP quiver field overlaid. To overplot AoLP quivers,
    Stokes Q, U and I must be provided. The quivers are oriented by AoLP and scaled by
    polarization fraction (estimated as image/I on binned maps).
    
    The function:
      1. Displays the image in pixel units or milliarcseconds (if `ps_mas` is given),
      2. Optionally rois to a square region of interest (ROI),
      3. Optionally bins the image (and Stokes maps, if provided) by integer factors,
      4. Optionally overlays quivers oriented by AoLP and scaled by polarization fraction,
      5. Optionally applies SNR masking for quiver vectors,
      6. Optionally outlines the mask region with contours.


    Parameters
    ----------
    image_to_plot : np.ndarray
        2D array to use as the background image (e.g. Q_phi, polarized intensity, or total intensity).
    ps_mas : float or None
        Pixel scale in milliarcseconds per pixel. If None, axes are shown in pixel units.
    image_scale : {"linear", "log", "asinh"}, default="linear"
        Scaling applied to the background image:
          * "linear" : raw values
          * "log"    : logarithmic scaling (values <= 0 will be masked)
          * "asinh"  : inverse hyperbolic sine stretch, useful for high dynamic range
    title : str, optional
        Title for the plot.
    save : str or None, optional
        Path to save the figure. If None, the figure is not saved.
    show : bool, default=True
        If True, display the figure with `plt.show()`.
    roi_half_size : int or None, optional
        Half-size of the square ROI. If provided, roi to a region
        of size (2*roi_half_size) × (2*roi_half_size) around `roi_center`.
    roi_center : (int, int) or None, optional
        Center (y, x) of the ROI. If None, the image center is used.
    cmap_image : str, default="viridis"
        Colormap for the background image.
    cbar_label : str, optional
        Label for the colorbar.
    return_fig_ax : bool, default=False
        If True, return (fig, ax) for further modification.
    aolp_quiver : bool, default=False
        If True, overlay AoLP quivers computed from Stokes Q and U.
        Requires Q, U, and I to be provided.
    bin_factor : (int, int), default=(4, 4)
        Integer binning factors (by_y, by_x) for plotting the AoLP quivers. 
        Must evenly divide the roiped image dimensions.
    Q, U, I : np.ndarray or None, optional
        Stokes Q, U, and total intensity maps. Needed only if `aolp_quiver=True`.
    snr_threshold : float or None, optional
        SNR threshold for masking quivers. Vectors are kept where
        image_to_plot >= snr_threshold * noise_level. Requires `noise_level`.
    noise_level : float or None, optional
        Noise estimate in the same units as `image_to_plot`. Used with `snr_threshold`.
    quiver_scale : float or None, default=0.1
        Scaling factor for quiver lengths passed to `ax.quiver`. If None, use Matplotlib defaults.
    Returns
    -------
    (fig, ax) or None
        Matplotlib Figure and Axes objects if `return_fig_ax=True`, otherwise None.

    Notes
    -----
    * AoLP is always recomputed from binned Q and U, not from pre-binned angles.
    * Polarization fraction is estimated as (image_to_plot / I) on binned data.
    * If `ps_mas` is given, axis labels are in mas; otherwise pixel coordinates are used.
    * This function is not yet fully verified.
    """

    # --- helpers ---
    def _center_crop(arr, half, center=None):
        if half is None:
            return arr
        ny, nx = arr.shape
        cy, cx = (ny // 2, nx // 2) if center is None else center
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        return arr[y0:y1, x0:x1]

    def _block_reduce_sum(a: np.ndarray, by: int, bx: int) -> np.ndarray:
        """Sum-reduce in (y,x) blocks of (by,bx). Assumes divisibility."""
        ny, nx = a.shape
        if ny % by != 0 or nx % bx != 0:
            raise ValueError(f"Array shape {a.shape} not divisible by bin_factor {(by, bx)}")
        return a.reshape(ny // by, by, nx // bx, bx).sum(axis=(1, 3))
    
    ################################################################################
    # --- input checks ---
   
    allowed_scales = {"linear", "log", "asinh"}
    if image_scale not in allowed_scales:
        raise ValueError(
            f"Invalid image_scale '{image_scale}'. "
            f"Choose one of {allowed_scales}."
        )
    
    if aolp_quiver:
        if Q is None or U is None or I is None:
            raise ValueError("To plot AoLP quivers, Stokes Q, U and I must be provided.")
        if not (Q.shape == U.shape == I.shape == image_to_plot.shape):
            raise ValueError("All input maps must have the same shape.")

    # ---   p (optional) ---
    if roi_half_size is not None:
        image_to_plot = _center_crop(image_to_plot, roi_half_size, roi_center)
        if aolp_quiver:
            Q = _center_crop(Q, roi_half_size, roi_center)
            U = _center_crop(U, roi_half_size, roi_center)
            I = _center_crop(I, roi_half_size, roi_center)



    ny, nx = image_to_plot.shape
    
    # --- figure/axes ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # --- extent in mas or pixels ---
    if ps_mas is not None:
        half_w_mas = (nx * ps_mas) / 2.0
        half_h_mas = (ny * ps_mas) / 2.0
        extent = (-half_w_mas, half_w_mas, -half_h_mas, half_h_mas)  # (x_min, x_max, y_min, y_max)
        xlabel, ylabel = "mas", "mas"
    else:
        extent = (-(nx / 2), (nx / 2), -(ny / 2), (ny / 2))
        xlabel, ylabel = "pixel", "pixel"

    # --- background plot ---
    if image_scale == "linear":
        img_display = image_to_plot
        if cbar_label is None:
            cbar_label = "Intensity (linear)"
    elif image_scale == "log":
        img_display = np.where(image_to_plot > 0, np.log10(image_to_plot), np.nan)
        if cbar_label is None:
            cbar_label = "Intensity (log10)"
    elif image_scale == "asinh":
        img_display = np.arcsinh(image_to_plot)
        if cbar_label is None:
            cbar_label = "Intensity (asinh)"
    
    im = ax.imshow(img_display, origin="lower", cmap=cmap_image, extent=extent, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.73)
    cbar.set_label(cbar_label)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    
    # --- quiver overlay (optional) ---
    if aolp_quiver:
        by, bx = bin_factor
        
        # --- bin maps (sum) ---
        # Use sums for Q/U/I so that AoLP is computed from vector sums; fraction computed from binned Qphi/I.
        image_to_plot_b = _block_reduce_sum(image_to_plot, by, bx)
        if aolp_quiver:
            Q_b = _block_reduce_sum(Q, by, bx)
            U_b = _block_reduce_sum(U, by, bx)
            I_b = _block_reduce_sum(I, by, bx)
    
        ny_b, nx_b = image_to_plot_b.shape

        # --- coordinates for quiver centers in the same units as the extent ---
        if ps_mas is not None:
            xs = np.linspace(-half_w_mas + (ps_mas * bx) / 2, half_w_mas - (ps_mas * bx) / 2, nx_b)
            ys = np.linspace(-half_h_mas + (ps_mas * by) / 2, half_h_mas - (ps_mas * by) / 2, ny_b)
        else:
            xs = np.linspace(-(nx / 2) + bx / 2, (nx / 2) - bx / 2, nx_b)
            ys = np.linspace(-(ny / 2) + by / 2, (ny / 2) - by / 2, ny_b)

        Xc, Yc = np.meshgrid(xs, ys)

        
        # --- recompute AoLP (psi) from binned Q/U ---
        psi = 0.5 * np.arctan2(U_b, Q_b)  # radians

        # --- polarization fraction (simple estimator) ---
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(I_b != 0.0, image_to_plot_b / I_b, 0.0)

        # --- masking ---
        mask = np.ones_like(image_to_plot_b, dtype=bool)
        if snr_threshold is not None:
            if noise_level is None:
                raise ValueError("`noise_level` must be provided when using `snr_threshold`.")
            mask &= (image_to_plot_b >= snr_threshold * noise_level)


        # Rotate by +pi/2 to follow usual convention (vectors start from North and increase to East).
        dx = frac * np.cos(psi + np.pi / 2.0)
        dy = frac * np.sin(psi + np.pi / 2.0)

        quiv_kwargs = dict(headlength=0, headwidth=1, pivot="middle", color="w")
        if quiver_scale is not None:
            quiv_kwargs["scale"] = quiver_scale

        ax.quiver(Xc[mask], Yc[mask], dx[mask], dy[mask], **quiv_kwargs)

        # optional: outline the mask
        try:
            # Build an integer mask for contouring (0/1)
            mask_int = mask.astype(int)
            ax.contour(
                Xc, Yc, mask_int,
                levels=[0.5], colors=["white"], linewidths=0.8
            )
        except Exception:
            pass

    ax.set_title(title, fontsize=16)
    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches="tight", pad_inches=0.1)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig_ax:
        return fig, ax
    return None


def central_crop(image: np.ndarray, half_size: int) -> np.ndarray:
    """Return a central crop with half-size (in pixels) around the image center.


    Parameters
    ----------
    image : ndarray
    2D image to crop.
    half_size : int
    Half-size of the square crop (total side = 2*half_size).


    Returns
    -------
    ndarray
    Cropped image.
    """
    n_y, n_x = image.shape
    cx, cy = n_x // 2, n_y // 2
    return image[cy - half_size : cy + half_size, cx - half_size : cx + half_size]


def compute_qphi_uphi_pi(q: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Qphi, UPhi and PI from Stokes Q and U using azimuthal angle of each pixel.
    Uses the standard Schmid 2021 convention with the local azimuthal angle
    phi = arctan2(Y, X) + pi/2  
    Qphi = -Q * cos(2*phi) - U * sin(2*phi)
    Uphi = Q * sin(2*phi) - U * cos(2*phi)
    Pi = sqrt(Q^2 + U^2)

    Parameters
    ----------
    q, u : ndarray
    Stokes Q and U images (same shape).


    Returns
    -------
    Qphi, Uphi, PI, phi : (ndarray, ndarray, ndarray, ndarray)
    Azimuthal Stokes images and the azimuth angle field used.
    """
    if q.shape != u.shape:
        raise ValueError("q and u must have the same shape")


    n = q.shape[0]
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q and u must be square 2D images")


    _,_,_,X, Y = compute_grid(q)
    phi = np.arctan2(Y, X) + np.pi / 2.0


    c = np.cos(2.0 * phi)
    s = np.sin(2.0 * phi)
    qphi = -q * c - u * s
    uphi = q * s - u * c
    pi= np.sqrt(q**2 + u**2)

    return qphi, uphi, pi, phi




def plot_image_grid(
    images: List[np.ndarray],
    ps_mas: float,
    *,
    nrows: int,
    ncols: int,
    titles: Optional[List[str]] = None,
    group_headers: Optional[List[Tuple[float, str]]] = None,
    scale: str = "linear",           # {"linear","log","asinh"}
    roi_half_size: Optional[int] = 30,  # in pixels, for autoscaling window; None = full frame
    roi_center: Optional[Tuple[int, int]] = None,  # (y,x) in pixels; None = image center
    per_panel_autoscale: bool = True,    # vmin/vmax from roi per image
    cmap: str = "viridis",
    colorbar: str = "none",  # {"shared","individual","none"}
    cbar_label: Optional[str] = None,
    cbar_kwargs: Optional[dict] = None,  # extra kwargs passed to color
    fontsize_axes: int = 14,
    fontsize_titles: int = 16,
    figsize: Tuple[float, float] = (12, 6),
    hide_axis_rules: Optional[callable] = None,  # function(ax, row, col) -> None to customise axis visibility
    tight: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Status: not fully verified

    Plot a grid of images with consistent angular axes (mas) and flexible intensity scaling.

    Parameters
    ----------
    images : list of 2D np.ndarray
        Images to plot; length must equal nrows * ncols. All images must share the same shape.
    ps_mas : float
        Pixel scale in milliarcseconds per pixel.
    nrows, ncols : int
        Grid layout.
    titles : list of str, optional
        Per-panel titles; length should be <= len(images). Extra panels ignore titles.
    group_headers : list of (x_position, text), optional
        Figure-level headers placed at fig coords (x, 1.0). Useful for labeling column groups.
    scale : {"linear","log","asinh"}
        Intensity transform applied for display.
    roi_half_size : int or None, optional
        Half-size of the square ROI. If provided, crop to a region
        of size (2*roi_half_size) × (2*roi_half_size) around `roi_center`.
    roi_center : (int, int) or None, optional
        Center (y, x) of the ROI. If None, the image center is used.
    per_panel_autoscale : bool
        If True, compute vmin/vmax per panel; otherwise use global vmin/vmax from first image (after transform).
    cmap : str
        Colormap for imshow.
    colorbar : {"shared","individual","none"}
        Add one shared colorbar, one per subplot, or none.
    cbar_label : str or None
        Label for the colorbar(s).
    cbar_kwargs : dict, optional
        Extra kwargs passed to colorbar (e.g., dict(fraction=0.02, pad=0.02, shrink=0.9))
    fontsize_axes, fontsize_titles : int
        Font sizes for axes and titles.
    figsize : (float, float)
        Figure size in inches.
    hide_axis_rules : callable or None
        Optional hook called as hide_axis_rules(ax, row, col) to customise which spines/ticks/labels are shown.
    tight : bool
        If True, apply tight_layout().
    show : bool
        If True, call plt.show().

    Returns
    -------
    fig, axs : matplotlib Figure and Axes array.
    """

    # --- validate
    if len(images) != nrows * ncols:
        raise ValueError(f"Expected {nrows*ncols} images, got {len(images)}.")
    shapes = {im.shape for im in images}
    if len(shapes) != 1:
        raise ValueError("All images must have the same shape.")
    
    
    # --- choose transform
    allowed_scales = {"linear", "log", "asinh"}
    if scale not in allowed_scales:
        raise ValueError(f"scale must be one of {allowed_scales}, got '{scale}'.")

    def transform(img: np.ndarray) -> np.ndarray:
        if scale == "log":
            # mask non-positive values for log display
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.full_like(img, np.nan, dtype=float)
                pos = img > 0
                out[pos] = np.log10(img[pos])
            return out
        if scale == "asinh":
             return np.arcsinh(img)
        return img

    # --- extent in mas (keep your original orientation: extent=(-d, d, d, -d))
    ny, nx = images[0].shape
    if roi_half_size:
        d=roi_half_size * ps_mas
    else:
        d = (np.max([nx, ny]) - 1) * ps_mas / 2.0

    extent = (-d, d, d, -d)

    # --- helper for central crop min/max
    def crop_minmax(img_t: np.ndarray) -> Tuple[float, float]:
        if roi_half_size is None:
            sub = img_t
        else:
            if roi_center is not None:
                cy, cx = roi_center
            else:
                cy, cx = ny // 2, nx // 2
            y0, y1 = int(cy - roi_half_size/2), int(cy + roi_half_size/2)
            x0, x1 = int(cx - roi_half_size/2), int(cx + roi_half_size/2)
            sub = img_t[y0:y1, x0:x1]
        # robust min/max even if NaNs present
        return (np.nanmin(sub), np.nanmax(sub))
    
    # --- precompute transforms and min/max
    images_t = [transform(im) for im in images]

    if per_panel_autoscale:
        minmax = [crop_minmax(imt) for imt in images_t]
    else:
        rois = [crop_minmax(imt) for imt in images_t]
        global_vmin = np.min([mn for (mn, mx) in rois])
        global_vmax = np.max([mx for (mn, mx) in rois])
        minmax = [(global_vmin, global_vmax) for _ in images_t]

    # --- plot
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = np.atleast_2d(axs)

    if hide_axis_rules is None:
        def hide_axis_rules(ax, r, c):
            # Example: hide x labels on top row; hide y labels on right 3 panels per row
            if r == 0:
                ax.get_xaxis().set_visible(False)
            if c > 0:
                ax.get_yaxis().set_visible(False)

    im_handles = []  # store imshow handles for colorbars
    for idx, (ax, imt, (vmin, vmax)) in enumerate(zip(axs.flat, images_t, minmax)):
        im = ax.imshow(imt, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap)
        im_handles.append(im)
        ax.set_xlim(-d,d)
        ax.set_ylim( d, -d)  
        ax.set_xlabel("mas", fontsize=fontsize_axes)
        ax.set_ylabel("mas", fontsize=fontsize_axes)
        ax.tick_params(axis='both', labelsize=fontsize_axes)
        r, c = divmod(idx, ncols)
        hide_axis_rules(ax, r, c)
        if titles and idx < len(titles) and titles[idx]:
            ax.set_title(titles[idx], fontsize=fontsize_titles)
        
    # --- colorbars
    cbar_kwargs = cbar_kwargs or {}
    if colorbar == "shared":
        if per_panel_autoscale:
            print("Warning: shared colorbar with per_panel_autoscale=True corresponds only to last image.")
        # compute the tight bounding box of all subplots
        boxes = np.array([ax.get_position().extents for ax in axs.ravel()])
        left, bottom = boxes[:, 0].min(), boxes[:, 1].min()
        right, top   = boxes[:, 2].max(), boxes[:, 3].max()

        # space for colorbar to the right of the whole grid
        cbar_pad   = 0.01   # gap between grid and colorbar (figure coords)
        cbar_width = 0.02   # colorbar width (figure coords)
        cbar_ax = fig.add_axes([right + cbar_pad, bottom, cbar_width, top - bottom])

        # one shared colorbar (use any image handle; they share the same cmap & norm if global scale is used)
        im_ref = im_handles[-1]
        cbar = fig.colorbar(im_ref, cax=cbar_ax) 
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=fontsize_axes)
        cbar.ax.tick_params(labelsize=fontsize_axes)
        
    elif colorbar == "individual":
        # one colorbar per axes
        for ax, im in zip(axs.flat, im_handles):
            cbar = plt.colorbar(im, ax=ax, **({"orientation": "vertical", "fraction": 0.046, "pad": 0.04} | cbar_kwargs))
            if cbar_label: cbar.set_label(cbar_label, fontsize=fontsize_axes-2)
            cbar.ax.tick_params(labelsize=fontsize_axes-2)
    elif colorbar == "none":
        pass
    else:
        raise ValueError("colorbar must be one of {'shared','individual','none'}")

    # group headers at top (figure coords)
    if group_headers:
        for x, text in group_headers:
            if colorbar == "shared":
                fig.text(x, .95, text, fontsize=fontsize_titles, ha='center', va='bottom')
            else:
                fig.text(x, 1.0, text, fontsize=fontsize_titles, ha='center', va='bottom')

    if tight and colorbar != "shared":
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=0.01)
    if show:
        plt.show(fig)
    plt.close(fig)

    return fig, axs

def rescale_and_recalculate_all_polarim_img(
    img_q: np.ndarray,
    img_u: np.ndarray,
    img_tot: np.ndarray,
    current_pix_scale: float,
    instrument: Optional[str] = None,
    new_pix_scale: Optional[float] = None,
    *,
    conserve: str = "surface_brightness",  # or "sum"
    order: int = 3,                        # cubic interpolation
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample Q, U, and total images to a new pixel scale and recompute q_phi, u_phi, PI, phi.

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
    img_q_res, img_u_res, img_tot_res, q_phi_res, u_phi_res, pi_res, phi_res
    """
    if new_pix_scale is None:
        if instrument is None:
            raise ValueError("Provide either `instrument` or `new_pix_scale`.")
        inst = instrument.lower()
        if inst == "irdis":
            new_pix_scale = IRDIS_PIX_MAS
        elif inst == "zimpol":
            new_pix_scale = ZIMPOL_PIX_MAS
        else:
            raise ValueError(f"Unknown instrument '{instrument}'. Use 'irdis' or 'zimpol', or pass new_pix_scale.")

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

    img_q_res = _res(img_q)
    img_u_res = _res(img_u)
    img_tot_res = _res(img_tot)

    # Optional intensity renormalization to conserve total flux
    if conserve == "sum":
        img_q_res /= s**2
        img_u_res /= s**2
        img_tot_res /= s**2
    elif conserve != "surface_brightness":
        raise ValueError("`conserve` must be 'surface_brightness' or 'sum'.")

    # If compute_qphi_uphi_pi needs a center, make sure it is handled consistently.
    # For example: compute_qphi_uphi_pi(..., cx = cx_old * s, cy = cy_old * s)
    q_phi_res, u_phi_res, pi_res, phi_res = compute_qphi_uphi_pi(img_q_res, img_u_res)

    return img_q_res, img_u_res, img_tot_res, q_phi_res, u_phi_res, pi_res, phi_res


def convolve_polarimetric_images(
    img_q: np.ndarray,
    img_u: np.ndarray,
    img_i: np.ndarray,
    ps: float, 
    *,
    psf_source: Literal["synthetic", "file"] = "synthetic",
    psf_fwhm_mas: Optional[float] = None,          # required if psf_source == "synthetic"
    psf_file: Optional[str] = None,                # Name of the PSF file, required if psf_source == "file"
    folder_psf: Optional[str]=None,                # FOlder where PSF file is, required if psf_source == "file"
    psf_array: Optional[np.ndarray] = None,      # direct PSF array input (overrides psf_file and folder_psf)
    psf_cut: Optional[int] = None,                 # half-size crop in pixels (file PSF)
    normalize_kernel: bool = True,                  #for convolve_fft
    boundary: Literal["fill", "wrap", "extend"] = "wrap",   #for convolve_fft
    nan_treatment: Literal["interpolate", "fill"] = "interpolate",  #for convolve_fft
    fft_pad: bool = False,                           #for convolve_fft
    allow_huge: bool = False,                       #for convolve_fft
    plot: bool = True,
    save: str = None, 
    roi_half_size: int = 30,
    image_scale: str = "asinh",
    silent: bool=False,
) -> Tuple[Any]:
    """
    Convolve Q, U, I with a PSF (synthetic or file-based) and compute Q_phi, U_phi, PI.

    Returns a tuple with convolved images.
    """
    
    if psf_source == "synthetic":
        if psf_fwhm_mas is None:
            raise ValueError("psf_fwhm_mas must be provided for synthetic PSF.")
        if not silent:
            print("Convolving with synthetic PSF")
            print(f"PSF FWHM: {psf_fwhm_mas} mas")
        psf = synthetic_psf(ps, psf_fwhm_mas)
    elif psf_source == "file":
        if psf_array is None  and (psf_file is None or folder_psf is None):
            raise ValueError("psf_array or psf_file and folder_psf must be provided for file PSF.")
        if not silent:
            print("Convolving with real PSF from observations")
        if psf_array is not None:
            psf = psf_array
        else:
            psf = Loadimage(folder_psf,psf_file)  # your loader should return a 2D array
    
    else:
        raise ValueError(f"Unknown psf_source '{psf_source}'")
    
    kernel=np.copy(psf)
    # Optional central crop to reduce ringing / speed up FFT
    if psf_cut is not None:
        n0, n1 = psf.shape
        c0, c1 = n0 // 2, n1 // 2
        kernel = psf[c0-psf_cut:c0+psf_cut, c1-psf_cut:c1+psf_cut]

    if normalize_kernel:
        s = kernel.sum()
        if s == 0 or not np.isfinite(s):
            raise ValueError("PSF kernel sum invalid (0 or non-finite).")
        kernel = kernel / s
    if not silent:
        print(f"PSF shape: {kernel.shape}, PSF sum: {kernel.sum()}")

    # --- Convolution (FFT) ---
    def _conv(x):
        return convolve_fft(
            np.asarray(x, dtype=float),
            kernel,
            boundary=boundary,
            nan_treatment=nan_treatment,
            normalize_kernel=False,   # we already normalized (or intentionally not) above
            allow_huge=allow_huge,
            preserve_nan=False,
            psf_pad=fft_pad,
            fft_pad=fft_pad,
        )

    Q_conv = _conv(img_q)
    U_conv = _conv(img_u)
    I_conv = _conv(img_i)
    if not silent:
        print(f"new shape after convolution: {Q_conv.shape}")

    # --- Derived products ---
    Q_phi_conv,U_phi_conv, PI_conv, phi =compute_qphi_uphi_pi(Q_conv, U_conv)
    
    
    # --- Optional plots ---
    if plot:
        plot_polarimetric_image(Q_conv,   ps, title="Q Conv",     roi_half_size=roi_half_size, image_scale=image_scale, save=save, show=True)
        plot_polarimetric_image(Q_phi_conv, ps, title="Q_phi Conv", roi_half_size=roi_half_size, image_scale=image_scale, save=save, show=True)
        plot_polarimetric_image(I_conv,   ps, title="I Conv",     roi_half_size=roi_half_size, image_scale=image_scale, save=save, show=True)
        plot_polarimetric_image(PI_conv,  ps, title="PI Conv",    roi_half_size=roi_half_size, image_scale=image_scale, save=save, show=True)

    return kernel, Q_conv, U_conv, I_conv, PI_conv, Q_phi_conv, U_phi_conv
        

def polarimetric_metrics(
    i: np.ndarray,
    q: np.ndarray, 
    u: np.ndarray, 
    q_phi: np.ndarray, 
    u_phi: np.ndarray, 
    pi: np.ndarray, 
    roi_half_size: Optional[int] = None,        # radius (px) for Qφ/Uφ/PI metrics
    roi_half_size_tot_int: Optional[int] = None,# radius (px) for I
    print_summary: bool = True
)-> Dict[str, Any]:
    """
    Compute integrated polarimetric metrics within a circular ROI.

    Returns dict with sums and fractions relative to total intensity.
    """
    R, _,_,_,_= compute_grid(i)

    if roi_half_size is None:
        mask = np.ones_like(i, dtype=bool)
    else:
        mask = R <= float(roi_half_size)

    if roi_half_size_tot_int is None:
        mask_i = mask
    else:
        mask_i = R <= float(roi_half_size_tot_int)

    # --- Metrics ---
    pi_sum = float(np.nansum(pi[mask]))
    i_sum  = float(np.nansum(i[mask_i]))
    q_sum = float(np.nansum(q[mask]))
    u_sum = float(np.nansum(u[mask]))
    qphi_sum = float(np.nansum(q_phi[mask]))
    uphi_sum = float(np.nansum(u_phi[mask]))

    pi_frac = pi_sum / i_sum if i_sum != 0 else np.nan
    qphi_frac = qphi_sum / i_sum if i_sum != 0 else np.nan
    uphi_frac = uphi_sum / i_sum if i_sum != 0 else np.nan
    q_phi_pi_frac = qphi_sum /pi_sum if pi_sum!=0 else np.nan


    out = {
        "sum_PI": pi_sum,
        "sum_I": i_sum,
        "sum_Q": q_sum,
        "sum_U": u_sum,
        "sum_Q_phi": qphi_sum,
        "sum_U_phi": uphi_sum,
        "frac_PI_over_I": pi_frac,
        "frac_Qphi_over_I": qphi_frac,
        "frac_Uphi_over_I": uphi_frac,
        "frac_Qphi_over_PI": q_phi_pi_frac,
        "roi_pixels": int(mask.sum()),
        "roi_pixels_I": int(mask_i.sum()),
        "roi_half_size": roi_half_size,
        "roi_half_size_tot_int": roi_half_size_tot_int,
    }

    if print_summary:
        print(f"sum Q_phi: {qphi_sum:.6g}, sum U_phi: {uphi_sum:.6g}, sum PI: {pi_sum:.6g}")
        if np.isfinite(pi_frac):   print(f"frac PI:   {100*pi_frac:.3f} %")
        if np.isfinite(qphi_frac): print(f"frac Qφ:   {100*qphi_frac:.3f} %")
        if np.isfinite(uphi_frac): print(f"frac Uφ:   {100*uphi_frac:.3f} %")

    return out 
    

def rotate_image(image, angle,xc,yc):
    """
    Rotate image around center (xc,yc) by angle in degrees. 
    positive angle rotates image conter-clockwise but we are plotting image upside down 
    (it is caused by extent parameter in the imshow) 
    and thats why it seems like positive angle corresponds to the clockwise direction. 
    In this case image looks similar to plotted with ds9 after irdap.  
    
    Parameters
    ----------
        image : 2D ndarray
            Input image to be rotated.
        angle : float
            Rotation angle in degrees. Positive values mean counter-clockwise rotation.
        xc : float
            X-coordinate of the rotation center.
        yc : float
            Y-coordinate of the rotation center.
    Returns
    -------
        2D ndarray
            Rotated image.
    """
    #angle in deg
    image_center = (xc,yc)
    (h, w) = image.shape[:2]

    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    result = cv2.warpAffine(image, rot_mat, (w, h))#,flags=cv2.INTER_LINEAR)

    return result


def radial_br_profile(
    img: np.ndarray,
    ps: float,                         # pixel scale (e.g., mas/px)
    inclination_deg: float,
    position_angle_deg: float,         # major-axis PA (deg)
    R_limit: float = 1000.0,           # limit in same units as ps (e.g., mas)
    *,
    noise_level: Optional[float] = None,     # fractional (err = noise_level * signal_mean)
    noise_map: Optional[np.ndarray] = None,  # per-pixel noise map (same shape as img)
    force_stop: Optional[bool] = False,    # stop radius in same units as ps (e.g., mas)
    plot: bool = True,
    save: Optional[str] = '',        # folder/prefix or None
    roi_half_size: int = 30,           # for the image plotting(pixels)
    image_scale: str = "asinh",
    mode: Literal["mean","median","sum"] = "mean", # how to compute brightness per annulus
    xc: Optional[float] = None,
    yc: Optional[float] = None,
    background_annulus_mas: Optional[tuple] = None,  # (rin, rout) in mas
) -> Dict[str, np.ndarray]:
    """

    Compute and plot a radial brightness profile with optional noise estimation.

    - Deprojection uses inclination + PA by rotating coordinates then compressing the minor axis by cos(i).
    - Binning uses annuli centered at radius i_r (in *pixels*) with half-width ~ sqrt(i_r)/2 pixels.
    - Error per bin is from: noise_map (std/sqrt(N) normalized), or noise_level * signal_mean, or a constant from background RMS.
    """
    if img.ndim != 2:
            raise ValueError("img must be 2D.")

    ny, nx = img.shape
    if xc is None or yc is None:
        xc = (nx - 1) / 2.0
        yc = (ny - 1) / 2.0

    
    inc = np.deg2rad(inclination_deg)
    cosi = max(1e-6, math.cos(inc)) #to avoid division by zero
    pa = math.radians(position_angle_deg)
    
    R,_,_,X, Y = compute_grid(img, xc=xc, yc=yc)
    #R_deproj_plot=np.sqrt(X**2 + (Y/cosi)**2)

    # Finite max and total intensity for normalization
    finite_max = np.nanmax(img[R< R_limit/ps])
    total_intensity = np.nansum(img[R< R_limit/ps])
    if not np.isfinite(finite_max) or finite_max == 0:
        raise ValueError("Image region set by upper radius (R_limit) has non-finite or zero maximum; cannot normalize.")
    
    # Apply PA rotation to coords, then deproject by cos(i)
    cos_t, sin_t = math.cos(-pa), math.sin(-pa)   # rotate coords by -PA so major axis aligns with x'
    Xp = cos_t * X - sin_t * Y
    Yp = sin_t * X + cos_t * Y

    R_deproj= np.sqrt(Xp**2 + (Yp/cosi)**2)   # in pixels
    
    # Optional noise map check
    if noise_map is not None:
        if noise_map.shape != img.shape:
            raise ValueError("noise_map shape must match img.")
        per_bin_noise = True
    else:
        per_bin_noise = False
    if background_annulus_mas is not None:
        rin_bg_mas, rout_bg_mas = background_annulus_mas
        rin_bg_px = rin_bg_mas / ps
        rout_bg_px = rout_bg_mas / ps
        mask_backg = (R< rout_bg_px) & (R >=rin_bg_px) & np.isfinite(img)
    else:
        mask_backg = (R < 460) & (R >=360) & np.isfinite(img)
    if np.any(mask_backg):

        backg=img[mask_backg]
        mad = np.nanmedian(np.abs(backg - np.nanmedian(backg))) # Median Absolute Deviation
        backgrms = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(backg) # robust std estimate
        back_mean=np.nanmean(backg) 
    else:
        backgrms=0.0
        back_mean=0.0
        
     # Accumulators
    errors, means, snrs = [], [], []

    img_rotated = rotate_image(img, position_angle_deg, xc,yc)

    # Iterate over radii in pixels up to R_limit
    max_i = int(max(2, math.floor(R_limit / ps)))
    used_r_px = []
    
    for i_r in range (2,max_i + 1,1):
        i_r_stop=i_r
        width_px = math.sqrt(i_r) / 2.0
        r_lo = (i_r - width_px)
        r_hi = (i_r + width_px)

        mask = (R_deproj < r_hi) & (R_deproj >= r_lo) & np.isfinite(img)
        nmask = int(np.count_nonzero(mask))
        if nmask == 0:
            continue
        
        vals = img[mask]
        if mode=="mean":
            signal_mean = float(np.nanmean(vals)) / finite_max
        elif mode=="median":
            signal_mean = float(np.nanmedian(vals)) / finite_max
        elif mode=="sum":
            signal_mean = float(np.nansum(vals)) / total_intensity

        
        if noise_map is not None:
            noise = noise_map[mask]
            
        # Error per bin
        if per_bin_noise:
            nvals = noise_map[mask]
            nsel= int(np.count_nonzero(np.isfinite(nvals)))
            err = float(np.nanstd(nvals)) / math.sqrt(nsel) / finite_max
        elif noise_level is not None:
            err = float(noise_level) * signal_mean
        elif (background_annulus_mas is not None) and (backgrms > 0):
            err = float(backgrms) / finite_max
        else:
            err = 0.0

        if (not force_stop) and (backgrms > 0) and (np.mean(signal_mean*finite_max)<(backgrms/2.0)): #this closely correspond to the snr contour
            break
        
       
        
        means.append(signal_mean)
        errors.append(err)
        snrs.append(signal_mean / max(err, 1e-16))  # avoid div by zero
        used_r_px.append(i_r)

    # Guard if no bins accumulated
    if not used_r_px:
        return {
            "i_rad_mas": np.array([]),
            "signal": np.array([]),
            "error": np.array([]),
            "snr": np.array([]),
        }

    i_rad_mas = np.array(used_r_px, dtype=float) * ps
    means = np.asarray(means)
    errors = np.asarray(errors)
    snrs = np.asarray(snrs)

    profile = {
        "i_rad_mas": i_rad_mas,
        "signal": means,
        "error": errors,
        "snr": snrs,
    }



    if plot:
        #plot profile+image
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6))
        ax1.errorbar(i_rad_mas,means, yerr = errors,ecolor='blue',color='black',fmt='o')
        
        # Reference for signal decreasing as r^-2
        # if means.size:
        #     k = int(np.argmax(means))
        #     r0 = i_rad_mas[k]
        #     if r0 > 0 and k < i_rad_mas.size - 1:
        #         ref = (np.max(means) / (i_rad_mas[k:] / r0) ** 2)
        #         ax1.plot(i_rad_mas[k:], ref, color="grey")
        
        ax1.set_ylabel('Normalised intensity', fontsize=24) 
        ax1.set_xlabel('Distance from the star, mas', fontsize=24)

        # Inset image 
        
        if image_scale == "linear":
            img_disp = img_rotated
        elif image_scale == "asinh":
            img_disp = np.arcsinh(img_rotated)
        else:
            raise ValueError("image_scale must be 'linear' or 'asinh'.")

        d = (img.shape[0] - 1) * ps / 2.0
        img2 = ax2.imshow(img_disp, vmax=np.nanmax(img_disp), extent=(-d, d, d / cosi, -d / cosi))    

        
        ax2.plot(xc*ps-d, yc*ps-d, "*", color='red')

        
        ax2.set_xlim(-roi_half_size*ps, roi_half_size*ps)
        ax2.set_ylim(-roi_half_size*ps, roi_half_size*ps)
        ax2.set_xlabel('mas', fontsize=24)
        ax2.set_ylabel('mas', fontsize=24)
        ax2.xaxis.set_tick_params(labelsize=20)
        ax2.yaxis.set_tick_params(labelsize=20)       
        ax1.xaxis.set_tick_params(labelsize=20)
        ax1.yaxis.set_tick_params(labelsize=20)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar=fig.colorbar(img2, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=20)
        plt.savefig(save+'radial_profile_linear+.jpeg',bbox_inches='tight', pad_inches=0.1)  
        
        ax1.set_yscale('log')
        ax1.set_ylabel('Normalised intensity, log units', fontsize=24)
        ax1.set_xlabel('Distance from the star, mas', fontsize=24)
        
        plt.savefig(save+'radial_profile_log+.jpeg',bbox_inches='tight', pad_inches=0.1) 
        #plt.show(block=True)
        plt.close(fig)

        
        # Plot just profile
        plt.errorbar(i_rad_mas, means, yerr=errors, ecolor="blue", color="black", fmt="o")
        if means.size:
            k = int(np.argmax(means))
            r0 = i_rad_mas[k]
            if r0 > 0 and k < i_rad_mas.size - 1:
                ref = (np.max(means) / (i_rad_mas[k:] / r0) ** 2)
                plt.plot(i_rad_mas[k:], ref, color="grey")
        plt.ylabel('Normalised intensity', fontsize=24) 
        plt.xlabel('Distance from the star, mas', fontsize=24)
        
        plt.savefig(save+ "radial_profile_linear.jpeg",bbox_inches='tight', pad_inches=0.1)
        plt.yscale('log')
        
        plt.savefig(save+'radial_profile_log.jpeg',bbox_inches='tight', pad_inches=0.1) 
        #plt.show(block=True)
        plt.close(fig)
    
    return profile

from matplotlib.patches import Circle
   
def azimuthal_profile(
    img: np.ndarray,
    ps: float,
    r_in_mas: float,
    r_out_mas: float,
    *,
    nbins: int = 180,
    xc: Optional[float] = None,
    yc: Optional[float] = None,
    mode: Literal["mean", "median", "sum"] = "sum",   #for calculating values per angle bin
    plot: bool = True,
    save: Optional[str] = None,
    theta0: Optional[float] = 0

) -> Dict[str, np.ndarray]:
    """
    Compute an azimuthal (position-angle) profile at a given annulus. 
    Parameters
    ----------
    img : 2D ndarray
        Input image.
    ps : float
        Pixel scale (e.g., mas/pix).
    r_in_mas : float
        Inner radius of the annulus (mas).
    r_out_mas : float
        Outer radius of the annulus (mas).
    nbins : int
        Number of angle bins (default: 180).
    xc : float, optional
        X-coordinate of the image center (default: image center).
    yc : float, optional
        Y-coordinate of the image center (default: image center).
    mode : {'mean','median','sum'}
        How to compute the value per angle bin:
        - 'mean': average intensity in the bin.
        - 'median': median intensity in the bin.
        - 'sum': total intensity in the bin.
        In all cases, the profile is normalized to the peak (mean/median) or total (sum) intensity in the annulus.
    plot : bool
        Whether to plot the azimuthal profile (default: True).
    save : str, optional
        Folder/prefix to save plots (if plot=True).
    theta0 : float
        Rotation angle in radians to define zero position angle direction (default: 0).

    Returns
    -------
    dict of np.ndarray
    Keys: ``theta_deg_centers``, ``value``, ``std``, ``stderr``, ``counts``,
    and ``theta_deg_edges``.
    """
    if img.ndim != 2:
        raise ValueError("img must be 2D")
    if mode not in ("mean", "median", "sum"):
        raise ValueError("Mode must be one of {'mean','median','sum'}")


    ny, nx = img.shape
    if xc is None or yc is None:
        xc = (nx - 1) / 2.0
        yc = (ny - 1) / 2.0
    R, _, _, X, Y = compute_grid(img, xc=xc, yc=yc)
    # Angle in radians in the rotated frame
    theta = np.arctan2(Y, X) # [-pi, pi]


        
    theta = theta - theta0
    # Map to [0, 2pi)
    theta = (theta + 2 * np.pi) % (2 * np.pi)


    # Annulus selection
    sel_ann = (R >= r_in_mas/ps) & (R < r_out_mas/ps) & np.isfinite(img)
    total_intensity=np.nansum(img[sel_ann])
    peak_intensity=np.nanmax(img[sel_ann])
    
    if not np.any(sel_ann):
        return {
        "theta_deg_centers": np.array([]),
        "value": np.array([]),
        "std": np.array([]),
        "stderr": np.array([]),
        "counts": np.array([]),
        "theta_deg_edges": np.array([]),
        }


    values = img[sel_ann]
    ang = theta[sel_ann]

    # Angle binning
    edges = np.linspace(0.0, 2 * np.pi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])


    value = np.full(nbins, np.nan)
    scat = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)


    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        sel_bin = (ang >= lo) & (ang < hi)
        counts[i] = int(np.count_nonzero(sel_bin))
        if counts[i] == 0:
            continue
        v = values[sel_bin]
        if mode=='median':
            value[i]= np.nanmedian(v)/peak_intensity
            scat[i] = np.nanstd(v)/peak_intensity
        elif mode=='mean':
            value[i] = np.nanmean(v)/peak_intensity
            scat[i] = np.nanstd(v)/peak_intensity
        elif mode=='sum':
            value[i] = np.nansum(v)/total_intensity
            scat[i] = np.nanstd(v)/total_intensity


    with np.errstate(divide="ignore", invalid="ignore"):
        stderr = scat / np.sqrt(np.maximum(counts, 1))


    theta_deg_centers = np.degrees(centers)
    theta_deg_edges = np.degrees(edges)


    out = {
    "theta_deg_centers": theta_deg_centers,
    "value": value,
    "std": scat,
    "stderr": stderr,
    "counts": counts,
    "theta_deg_edges": theta_deg_edges,
    }


    if plot:
        # ---------------- Figure 1: Cartesian profile + image overlay ----------------
        fig1, (ax_prof, ax_img) = plt.subplots(1, 2, figsize=(14, 5))

        # Profile
        ax_prof.errorbar(out["theta_deg_centers"], value, yerr=stderr, fmt="o", ms=3, lw=1)
        ax_prof.set_xlabel("Position angle [deg]")
        ax_prof.set_ylabel("Intensity [arb. units]")
        ax_prof.set_xlim(0, 360)
        ax_prof.grid(True, alpha=0.3)
        ax_prof.set_title("Azimuthal profile")

        # Image with highlighted annulus + θ0
        # Display in mas coordinates using extent
        d_x = (nx - 1) * ps / 2.0
        d_y = (ny - 1) * ps / 2.0
        img_disp = img  # linear display; adjust if you prefer asinh

        im = ax_img.imshow(
            img_disp,
            extent=(-d_x, d_x, d_y, -d_y),  # y inverted to match image coords
            vmax=np.nanmax(img_disp),
        )
        ax_img.set_xlabel("mas")
        ax_img.set_ylabel("mas")
        ax_img.set_title("Polarimetric image")

        # Draw annulus as two circles (in *pixels*, converted to mas via scale)
        r_in_px = r_in_mas / ps
        r_out_px = r_out_mas / ps

        # Convert center in pixels -> mas for plotting
        cx_mas = xc * ps - d_x
        cy_mas = yc * ps - d_y

        cin = Circle((cx_mas, cy_mas), radius=r_in_mas, fill=False, color="orange", lw=2)
        cout = Circle((cx_mas, cy_mas), radius=r_out_mas, fill=False, color="orange", lw=2, ls="--")
        ax_img.add_patch(cin)
        ax_img.add_patch(cout)

        # θ0 arrow: from center toward outer radius along θ0 (radians)
        theta0_rad = theta0
        x_end = cx_mas + r_out_mas * math.cos(theta0_rad)
        y_end = cy_mas + r_out_mas * math.sin(theta0_rad)
        ax_img.plot([cx_mas, x_end], [cy_mas, y_end], color="yellow", lw=2)
        ax_img.plot([x_end], [y_end], marker="o", color="yellow", ms=5)

        fig1.tight_layout()
            
        fig1.savefig(save + "azimuthal_profile_cartesian.jpeg", bbox_inches="tight", pad_inches=0.1)
        #plt.show()
        plt.close(fig1)

        # ---------------- Figure 2: Polar profile (separate figure) ----------------
        fig2 = plt.figure(figsize=(6, 6))
        ax_polar = fig2.add_subplot(111, projection="polar")
        ax_polar.errorbar(
            np.radians(out["theta_deg_centers"]), value, yerr=stderr, fmt="o", ms=3, lw=1
        )
        ax_polar.set_title("Azimuthal profile (Polar)")
        fig2.tight_layout()
        
        fig2.savefig(save + "azimuthal_profile_polar.jpeg", bbox_inches="tight", pad_inches=0.1)
        #plt.show()
        plt.close(fig2)
 


    return out




CAMERA_PS_MAS = {"zimpol": 3.6,      # mas/px
                 "irdis": 12.27     # mas/px
                }



def polarimetric_analysis(
    simulation_dir: str,
    wavelength: float,
    *,
    distance_pc: float = 1220.0, #important only to plot size in au and mas
    camera: Optional[Literal["zimpol", "irdis"]] = None,
    pixel_scale_mas: Optional[float] = None,
    convolution_mode: Literal["synthetic", "file", "none"] = "synthetic",
    psf_fwhm_mas: Optional[float] = None,       # required for synthetic (if camera not given)
    psf_file: Optional[str] = None,             # required for file mode
    psf_array: Optional[np.ndarray] = None,       # PSF array if convolution_mode == "file"
    folder_psf: Optional[str]=None,                # FOlder where PSF file is, required if psf_source == "file"
    psf_cut: Optional[int] = 100,               # optional crop for PSF
    image_scale: Literal["linear", "asinh"] = "asinh",
    unresolved_correction_radius_px: int = 3,   # radius (in px of rescaled image)
    background_annulus_mas: Optional[Tuple[float, float]] = (200, 250),
    radial_limit_mas: float = 150.0,
    deprojection: Tuple[float, float]=(0.0,0.0), # whether to deproject image for radial/azimuthal profiles, tuple of (incl_deg, pa_deg)
    azimuthal_r_in_mas: float = 0.0,
    azimuthal_r_out_mas: float = 100.0,
    azimuthal_nbins: int = 18,
    theta0: float = 0.0,                        # radians; 0 along +X
    plot: bool = True,
    roi_size_half: int = 30,
    fig_dir: Optional[str] = None,
    extra_title: Optional[str] = None,          # if provided, used as prefix for saved figures and title additions
) -> Dict[str, Any]:
        """
        Wrapper to produce polarimetric outputs from MCFOST model
        ----------
        Parameters

        """
        ### Safety checks
        wave_strs = str(wavelength)
        # locate RT.fits.gz 
        cand = simulation_dir +'/'+ f"data_{wave_strs}" +'/'+ "RT.fits.gz"
        rt_path = None
        if os.path.exists(cand):
                rt_path = cand
        if rt_path is None:
                raise FileNotFoundError("Could not find RT.fits.gz under in: "+ cand)

        # Output folder
        if fig_dir is None:
                fig_dir = str(simulation_dir +"/polarimetric")
        Path(fig_dir).mkdir(parents=True, exist_ok=True)


        if convolution_mode=='no':
                print('No convolution selected')
        elif convolution_mode=='synthetic':
                print('Synthetic PSF convolution selected')
                if psf_fwhm_mas is None:
                        if camera=='zimpol':
                                print ('PSF FWHM is 30 mas')
                                psf_fwhm_mas=30 #mas
                        elif camera=='irdis':
                                print ('PSF FWHM is 40 mas')
                                psf_fwhm_mas=40
                else:
                        print (f'PSF FWHM is {psf_fwhm_mas} mas')
        elif convolution_mode=='real':
                print('Real PSF convolution selected')
                if psf_file is None and psf_array is None:
                        raise ValueError('Please provide psf_file or psf_array for real PSF convolution')
                
        results = {}
        # --- Read header to get native angular pixel scale & FoV
        with fits.open(rt_path) as hdul:
                hdr = hdul[0].header

        # CDELT2 in degrees/pixel to mas/pixel
        native_ps_mas = float(hdr["CDELT2"]) * u.deg.to(u.arcsec) * 1000.0
        n_pix = int(hdr["NAXIS1"])
        fov_mas = n_pix * native_ps_mas
        fov_au=mas2au(fov_mas, distance_pc)
        
        
        # --- Choose instrument pixel scale
        if pixel_scale_mas is not None:
                inst_ps_mas = float(pixel_scale_mas)
        elif camera is not None:
                inst_ps_mas = CAMERA_PS_MAS[camera]
        else:
                # default to native if nothing specified
                inst_ps_mas = native_ps_mas

        # --- Load MCFOST images
        img_array_original, header_original, img_tot_original, img_q_original, img_u_original, img_v_original, img_star_original, img_star_sct_original, img_disk_th_original, img_disk_th_sct_original = load_mcfost_images_1wave(simulation_dir, wavelength, ploting=plot, save_plots=fig_dir, title_addition=extra_title)
        q_phi_original, u_phi_original, pi_original, phi_mcfost_original=compute_qphi_uphi_pi(img_q_original, img_u_original)
        metrics_original=polarimetric_metrics(img_tot_original,img_q_original, img_u_original, q_phi_original, u_phi_original,pi_original)

       
        results['mcfost_original']={'img_array':img_array_original, 
                                    'header':header_original, 
                                    'img_tot':img_tot_original, 
                                    'img_q':img_q_original, 
                                    'img_u':img_u_original, 
                                    'img_v':img_v_original, 
                                    'img_star':img_star_original, 
                                    'img_star_sct':img_star_sct_original, 
                                    'img_disk_th':img_disk_th_original, 
                                    'img_disk_th_sct':img_disk_th_sct_original,
                                    'q_phi':q_phi_original, 
                                    'u_phi':u_phi_original, 
                                    'pi':pi_original, 
                                    'phi':phi_mcfost_original,
                                    'metrics':metrics_original
                                    }

        # Rescaling to instrument pixel scale

        img_q_rescaled, img_u_rescaled, img_total_rescaled, q_phi_rescaled, u_phi_rescaled, pi_rescaled, phi_rescaled=rescale_and_recalculate_all_polarim_img(img_q_original, img_u_original, img_tot_original, native_ps_mas, new_pix_scale=inst_ps_mas, conserve='sum')
        metrics_rescaled=polarimetric_metrics(img_total_rescaled,img_q_rescaled, img_u_rescaled, q_phi_rescaled, u_phi_rescaled,pi_rescaled)

        results['mcfost_rescaled']={'img_q':img_q_rescaled,
                                        'img_u':img_u_rescaled,
                                        'img_tot':img_total_rescaled,
                                        'q_phi':q_phi_rescaled,
                                        'u_phi':u_phi_rescaled,
                                        'pi':pi_rescaled,
                                        'phi':phi_rescaled,
                                        'metrics':metrics_rescaled
                                        }

       

        # Convolving with synthetic PSF
        if convolution_mode=='synthetic':
                kernel, Q_conv, U_conv, I_conv, PI_conv, Q_phi_conv, U_phi_conv=convolve_polarimetric_images(img_q_rescaled, img_u_rescaled, img_total_rescaled, inst_ps_mas,psf_source='synthetic', psf_fwhm_mas=psf_fwhm_mas)
                metrics_conv=polarimetric_metrics(I_conv,Q_conv, U_conv, Q_phi_conv, U_phi_conv,PI_conv)

        if convolution_mode=='file':
                kernel, Q_conv, U_conv, I_conv, PI_conv, Q_phi_conv, U_phi_conv=convolve_polarimetric_images(img_q_rescaled, img_u_rescaled, img_total_rescaled, inst_ps_mas, psf_source='file', psf_file=psf_file, folder_psf=folder_psf,psf_array=psf_array, psf_cut=psf_cut )
                metrics_conv=polarimetric_metrics(I_conv,Q_conv, U_conv, Q_phi_conv, U_phi_conv,PI_conv)
        if convolution_mode=='none':
                Q_conv=img_q_rescaled
                U_conv=img_u_rescaled
                I_conv=img_total_rescaled
                PI_conv=pi_rescaled
                Q_phi_conv=q_phi_rescaled
                U_phi_conv=u_phi_rescaled
        if convolution_mode!='none':
                results['mcfost_convolved']={'img_q':Q_conv,
                                                'img_u':U_conv,
                                                'img_tot':I_conv,
                                                'q_phi':Q_phi_conv,
                                                'u_phi':U_phi_conv,
                                                'pi':PI_conv,
                                                'metrics':metrics_conv
                                                }



        if plot:
                plot_polarimetric_image(img_q_rescaled, inst_ps_mas, title='Q Rescaled', roi_half_size=roi_size_half, image_scale=image_scale, save=fig_dir, show=True)
                plot_polarimetric_image(q_phi_rescaled, inst_ps_mas, title='Q_phi Rescaled', roi_half_size=roi_size_half, image_scale=image_scale, save=fig_dir, show=True)

                plot_polarimetric_image(img_tot_original, native_ps_mas, title='I tot, original from mcfost', roi_half_size=roi_size_half, image_scale=image_scale, save=fig_dir, show=False)
                plot_polarimetric_image(img_total_rescaled, inst_ps_mas, title='I tot rescaled', roi_half_size=roi_size_half, image_scale=image_scale, save=fig_dir, show=False)
                plot_polarimetric_image(I_conv, inst_ps_mas, title='I tot conv', roi_half_size=roi_size_half, image_scale=image_scale, save=fig_dir, show=True)

        # Unresolved correction
        R_rescaled,_,_,_,_=compute_grid(img_q_rescaled)

        dolp_unres, aolp_unres,q_corr,u_corr=calculate_unresolved(unresolved_correction_radius_px, img_q_rescaled, img_u_rescaled,img_total_rescaled,inst_ps_mas,R_rescaled,100)
        q_phi_corr, u_phi_corr, pi_corr, phi =compute_qphi_uphi_pi(q_corr, u_corr)
        aolp_corr=0.5*np.arctan2(u_corr, q_corr)

        results['mcfost_not_convolved_unresolved_corrected']={'img_q':q_corr,
                                                'img_u':u_corr,
                                                'q_phi':q_phi_corr,
                                                'u_phi':u_phi_corr,
                                                'pi':pi_corr,
                                                'aolp_corr':aolp_corr,
                                                'dolp_unres':dolp_unres,
                                                'aolp_unres':aolp_unres,
                                                'phi':phi
                                                }
        
        # unresolved polarisation correction after convolution
        dolp_unres_conv, aolp_unres_conv,q_corr_conv,u_corr_conv=calculate_unresolved(unresolved_correction_radius_px, Q_conv, U_conv,I_conv,inst_ps_mas,R_rescaled,100)
        q_phi_corr_conv, u_phi_corr_conv, pi_corr_conv, phi =compute_qphi_uphi_pi(q_corr_conv, u_corr_conv)

        aolp_corr_conv=0.5*np.arctan2(u_corr_conv, q_corr_conv)
        
        results['mcfost_convolved_unresolved_corrected']={'img_q':q_corr_conv,
                                                'img_u':u_corr_conv,
                                                'q_phi':q_phi_corr_conv,
                                                'u_phi':u_phi_corr_conv,
                                                'pi':pi_corr_conv,
                                                'aolp_corr':aolp_corr_conv,
                                                'dolp_unres':dolp_unres_conv,
                                                'aolp_unres':aolp_unres_conv,
                                                'phi':phi
                                                }

        # print(f'Unresolved pol: {dolp_unres*100} %, angle: {aolp_unres} deg')

        # print(f'Unresolved pol after conv: {dolp_unres_conv*100} %, angle: {aolp_unres_conv} deg')
        if plot:
                images_list = [q_phi_rescaled, pi_rescaled, q_phi_corr, pi_corr,
                        Q_phi_conv, PI_conv, q_phi_corr_conv, pi_corr_conv
                        ]
                
                titles = ['Q$_\\phi$', 'I$_{\\mathrm{pol}}$', 'Q$_\\phi$', 'I$_{\\mathrm{pol}}$',
                        'Q$_\\phi$', 'I$_{\\mathrm{pol}}$', 'Q$_\\phi$', 'I$_{\\mathrm{pol}}$']

        # print(ps, type(ps))
        
                fig, axs = plot_image_grid(
                                images=images_list,
                                ps_mas=inst_ps_mas,
                                nrows=2,
                                ncols=4,
                                titles=titles,
                                group_headers=[(0.31, 'With unresolved'), (0.72, 'Without unresolved')],
                                scale="asinh",
                                roi_half_size=roi_size_half,          
                                per_panel_autoscale=True,
                                colorbar="individual",
                                figsize=(12, 6),
                                show=True
                                )
                fig.savefig(fig_dir+"mcfost_model_comparison.png", dpi=150, bbox_inches='tight')



        # kf.plot_polarimetric_image(q_phi_corr_conv,ps,Q=q_corr_conv,U=u_corr_conv,I=I_conv,title="convolved, unresolved corrected Q phi",bin_factor=(4,4),save=False,snr_threshold=3,noise_level=5e-19,roi_half_size=30,aolp_quiver=True, quiver_scale=0.1)
        # kf.plot_polarimetric_image(pi_rescaled,ps,Q=img_q_rescaled,U=img_u_rescaled,I=img_total_rescaled,title="Q phi",bin_factor=(4,4),save=False,snr_threshold=3,noise_level=2e-17,roi_half_size=30,aolp_quiver=True, quiver_scale=5)
        # kf.plot_polarimetric_image(q_phi_corr_conv, ps, roi_half_size=30, image_scale="asinh")
        # kf.plot_polarimetric_image(Q_conv, ps, roi_half_size=30, image_scale="linear")
        # kf.plot_polarimetric_image(I_conv, ps, roi_half_size=30, image_scale="linear")

        # kf.plot_polarimetric_image(q_phi,pixel_scale,Q=img_q,U=img_u,I=img_tot,title="Q phi",bin_factor=(4,4),image_scale="asinh",save=False,snr_threshold=3,noise_level=1e-17,roi_half_size=100,aolp_quiver=True, quiver_scale=3)



        radial_profile=radial_br_profile(PI_conv, inst_ps_mas,deprojection[0],deprojection[1], R_limit=radial_limit_mas/inst_ps_mas, mode='sum',save=fig_dir+"mcfost_", plot=True,background_annulus_mas=background_annulus_mas)


        az_profile=azimuthal_profile(PI_conv, inst_ps_mas, r_in_mas=azimuthal_r_in_mas, r_out_mas=azimuthal_r_out_mas, plot=True,mode='sum', save=fig_dir+"mcfost_", nbins=azimuthal_nbins, theta0=theta0)

        results['radial_profile']=radial_profile
        results['azimuthal_profile']=az_profile

        return results









def profiles_chi2(
    obs_data: np.ndarray,
    model_data: np.ndarray,
    ps: float,
    *,
    obs_err: Optional[np.ndarray]=None,
    noise_level: Optional[float] = None,
    profile_type: Literal["radial", "azimuthal", "both"] = "radial",
    mode: Literal["mean", "median", "sum"] = "mean",
    plot: bool = False,
    save: Optional[str] = '',
    deprojection_inc_pa_deg: Optional[Tuple[float, float]] = None,
    center: Optional[Tuple[float, float]] = None,
    az_nbins: int = 20,
) -> Tuple[float, float, float, int]:
    """
    Calculate the reduced chi2 between data and model contained in arrays.

    Parameters
    ----------
    obs_data : ndarray
        Observed data array (pi image, q_phi image, etc).
    model_data : ndarray
        Model data array.
    ps : float
        Pixel scale (e.g., mas/pix).
    obs_err : ndarray
        Observational error array (same shape as obs_data, uphi typically).
    noise_level : float
        Fractional noise level (err = noise_level * signal_mean) if obs_err is None
    deprojection_inc_pa_deg : tuple of float, optional
        (inclination_deg, position_angle_deg) for deprojection. If None, no deprojection is applied.
    center : tuple of float, optional
        (xc, yc) center coordinates in pixels. If None, image center is used.
    profile_type : {'radial','azimuthal','both'}
        Type of profile to compute chi2 on.
    mode : {'mean','median','sum'}
        How to compute profile values per bin.
    plot : bool
        Whether to plot profiles.
    save : str, optional
        Folder/prefix to save plots (if plot=True).

    Returns
    -------
    Tuple containing (chi2, chi2_red, loglike, n_data_points)
     """
    
    # Initialize chi2 accumulators
    chi2_sum = 0.0
    loglike_sum=0.0
    n_data_points = 0

    if deprojection_inc_pa_deg is not None:
        inc_deg, pa_deg = deprojection_inc_pa_deg
    else:
        inc_deg, pa_deg = 0.0, 0.0
    
    if center is not None:
        xc=center[0]
        yc=center[1]
    else:
        xc=None
        yc=None

    chi2_sum_radial = 0.
    loglike_sum_radial = 0.
    n_points_radial=0
    # Compute profiles
    if profile_type in ("radial", "both"):
        prof_obs = radial_br_profile(obs_data, ps, inclination_deg=inc_deg, position_angle_deg=pa_deg,
                                     mode=mode, noise_map=obs_err, noise_level=noise_level,xc=xc,yc=yc,
                                     plot=plot, save=save+"obs_")
        R_limit= np.max(prof_obs["i_rad_mas"])

        prof_mod = radial_br_profile(model_data, ps, inclination_deg=inc_deg, position_angle_deg=pa_deg,
                                     R_limit=R_limit,force_stop=True, mode=mode, xc=xc,yc=yc,
                                     plot=plot, save=save+"model_")
        
        chi2_sum_radial= ((prof_obs["signal"] - prof_mod["signal"]) ** 2 / (prof_obs["error"] ** 2 + 1e-16)).sum()
        loglike_sum_radial = np.nansum(((prof_obs["signal"] - prof_mod["signal"]) ** 2)/(prof_obs["error"] ** 2 + 1e-16) + np.log(2.0 * np.pi * (prof_obs["std"] ** 2 + 1e-16)))
        n_points_radial=len(prof_obs["signal"])
    
    if profile_type in ("azimuthal", "both"):
        r_in_mas = 0
        if profile_type=="azimuthal":
            r_out_mas= 1000.0
        else:
            r_out_mas =R_limit

        prof_obs_az = azimuthal_profile(obs_data, ps, r_in_mas, r_out_mas,
                                       mode=mode, xc=xc, yc=yc, nbins=az_nbins,
                                       plot=plot, save=save+"obs_")
        prof_mod_az = azimuthal_profile(model_data, ps, r_in_mas, r_out_mas,
                                       mode=mode, xc=xc, yc=yc, nbins=az_nbins,
                                       plot=plot, save=save+"model_")
        
        chi2_sum_az = ((prof_obs_az["value"] - prof_mod_az["value"]) ** 2 / (prof_obs_az["std"] ** 2 + 1e-16)).sum()
        loglike_sum_az = np.nansum(((prof_obs_az["value"] - prof_mod_az["value"]) ** 2)/(prof_obs_az["std"] ** 2 + 1e-16) + np.log(2.0 * np.pi * (prof_obs_az["std"] ** 2 + 1e-16)))
        n_points_az=len(prof_obs_az["value"])
       
    # Combine results
    chi2_sum = chi2_sum_radial + (chi2_sum_az if profile_type in ("azimuthal", "both") else 0.0)
    loglike_sum = loglike_sum_radial + (loglike_sum_az if profile_type in ("azimuthal", "both") else 0.0)
    n_data_points = n_points_radial + (n_points_az if profile_type in ("azimuthal", "both") else 0)

   
    
    if n_data_points == 0:
        raise ValueError("No valid data points found for chi2 calculation.")
    chi2_red = chi2_sum / (n_data_points-1)
    loglike=-0.5*loglike_sum
    
    return chi2_sum, chi2_red, loglike, n_data_points














#########################################
# MCFOST
##########################################

class ParaFile:
    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = []
        self.params = {}
        self.read()
        self._param_map = {
            "nbr_photons_eq_th": (3,0),
            "nbr_photons_lambda": (4,0),
            "nbr_photons_image": (5,0),
            "n_lambda": (8, 0),
            "lambda_min": (8, 1),
            "lambda_max": (8, 2),
            "compute_temp": (9, 0),
            "compute_sed": (9, 1),
            "use_default_lambda_grid": (9, 2),
            "wavelength_file": (10, 0),
            "separation_of_contributions":(11,0),
            "stokes parameters": (11, 1),
            "grid_geometry": (14, 0),
            "n_rad": (15, 0),
            "nz": (15, 1),
            "n_az": (15, 2),
            "n_rad_in": (15, 3),
            "grid_nx": (18, 0),
            "grid_ny": (18, 1),
            "grid_size": (18, 2),
            "imin": (19, 0),
            "imax": (19, 1),
            "n_incl": (19, 2),
            "centered": (19, 3),
            "az_min": (20, 0),
            "az_max": (20, 1),
            "n_az": (20, 2),
            "distance_pc": (21,0),
            "disk_pa": (22, 0),
            "scattering_mode": (25, 0),
            "image_symmetry": (28, 0),
            "central_symmetry": (29, 0),
            "axial_symmetry": (30, 0),
            "dust_settling": (33, 0),
            "exp_strat": (33, 1),
            "a_strat": (33, 2),
            "dust_radial_migration": (34, 0),
            "sublimate_dust": (35, 0),
            "hydrostatic_equilibrium": (36, 0),
            "viscous_heating": (37, 0),
            "alpha_viscosity": (37, 1),
            "number_of_zones": (40, 0)
        }
        num_zones = int(self.lines[self.find_line_starting_with("#Number of zones") + 1].split()[0])
        
        # Add density structure parameters for each zone
        start_line = self.find_line_starting_with("#Density structure")  # First zone starts 
        if start_line != -1:
            lines_per_zone = 7
            for i in range(num_zones):
                base = start_line +1+ i * lines_per_zone
                self._param_map[f"zone_{i+1}_type"] = (base + 0, 0)
                self._param_map[f"zone_{i+1}_dust_mass"] = (base + 1, 0)
                self._param_map[f"zone_{i+1}_gas_to_dust"] = (base + 1, 1)
                self._param_map[f"zone_{i+1}_scale_height"] = (base + 2, 0)
                self._param_map[f"zone_{i+1}_Rref"] = (base + 2, 1)
                self._param_map[f"zone_{i+1}_vertical_profile_exponent"] = (base + 2, 2)
                self._param_map[f"zone_{i+1}_Rin"] = (base + 3, 0)
                self._param_map[f"zone_{i+1}_edge"] = (base + 3, 1)
                self._param_map[f"zone_{i+1}_Rout"] = (base + 3, 2)
                self._param_map[f"zone_{i+1}_Rc"] = (base + 3, 3)
                self._param_map[f"zone_{i+1}_flaring_exp"] = (base +4,0)
                self._param_map[f"zone_{i+1}_surface_density_exp"] = (base +5,0)
                self._param_map[f"zone_{i+1}_-gamma_exp"] = (base +5,1)

        # Add grain properties for each zone
        start_line = self.find_line_starting_with("#Grain properties") 
        if start_line != -1:
            
            base = start_line + 1
            lines_per_species = 4
            for i in range(num_zones):
                self._param_map[f"zone_{i+1}_number_of_species"] = (base + 0, 0)
                num_species = int(self.lines[base + 0].split()[0])
                for j in range(num_species):
                    basej = base 
                    self._param_map[f"zone_{i+1}_species_{j+1}_grain_type"] = (basej + 1, 0)
                    self._param_map[f"zone_{i+1}_species_{j+1}_N_components"] = (basej + 1, 1)
                    self._param_map[f"zone_{i+1}_species_{j+1}_mixing_rule"] = (basej + 1, 2)
                    self._param_map[f"zone_{i+1}_species_{j+1}_porosity"] = (basej + 1, 3)
                    self._param_map[f"zone_{i+1}_species_{j+1}_mass_fraction"] = (basej + 1, 4)
                    self._param_map[f"zone_{i+1}_species_{j+1}_Vmax"] = (basej + 1, 5)
                    self._param_map[f"zone_{i+1}_species_{j+1}_optical_indices_file"] = (basej + 2, 0)
                    self._param_map[f"zone_{i+1}_species_{j+1}_volume_fraction"] = (basej + 2, 1)
                    self._param_map[f"zone_{i+1}_species_{j+1}_heating_method"] = (basej + 3, 0)
                    self._param_map[f"zone_{i+1}_species_{j+1}_amin"] = (basej +4,0)
                    self._param_map[f"zone_{i+1}_species_{j+1}_amax"] = (basej +4,1)
                    self._param_map[f"zone_{i+1}_species_{j+1}_aexp"] = (basej +4,2)
                    self._param_map[f"zone_{i+1}_species_{j+1}_n_grains"] = (basej +4,3)
                base = basej+4 +2
                
        
        #Add star properties
        start_line= self.find_line_starting_with("#Star properties")
        num_stars = int(self.lines[self.find_line_starting_with("#Star properties") + 1].split()[0])
        self._param_map["number_of_stars"] = (start_line + 1, 0)
        # Add  parameters for each star
        if start_line != -1:
            lines_per_star = 4
            for i in range(num_stars):
                base = start_line +2+ i * lines_per_star
                self._param_map[f"star_{i+1}_Temp"] = (base + 0, 0)
                self._param_map[f"star_{i+1}_R"] = (base + 0, 1)
                self._param_map[f"star_{i+1}_M"] = (base + 0, 2)
                self._param_map[f"star_{i+1}_x"] = (base + 0, 3)
                self._param_map[f"star_{i+1}_y"] = (base + 0, 4)
                self._param_map[f"star_{i+1}_z"] = (base + 0, 5)
                self._param_map[f"star_{i+1}_autometic_spectrum"] = (base +0, 6)
                self._param_map[f"star_{i+1}_spectrum_file"] = (base + 1, 0)
                self._param_map[f"star_{i+1}_fUV"] = (base + 2, 0)
                self._param_map[f"star_{i+1}_slope_fUV"] = (base + 2, 1)
                
        
        self._extract_params()

        


    def read(self):
        with open(self.filepath, "r") as f:
            self.lines = f.readlines()
        

    def _extract_params(self):
        for name, (line_no, col_no) in self._param_map.items():
            line = self.lines[line_no].strip()
            # print(line.split())
            value = line.split()[col_no]
            self.params[name] = value
   
    def find_line_starting_with(self, prefix):
        for i, line in enumerate(self.lines):
            if line.strip().startswith(prefix):
                return i
        return -1  # not found
        


    def set_param(self, param_name, new_value):
        if param_name not in self._param_map:
            raise ValueError(f"Unknown parameter name: {param_name}")
        line_no = self._param_map[param_name][0]
        col_no = self._param_map[param_name][1]
        old_line = self.lines[line_no]
        parts = old_line.split()
        parts[col_no] = str(new_value)
        self.lines[line_no] = "  "+"  ".join(parts) + "\n"
        self.params[param_name] = new_value

    def save(self, out_path):
        with open(out_path, "w") as f:
            f.writelines(self.lines)

    def print_params(self):
        for k, v in self.params.items():
            print(f"{k}: {v}")



def run_mcfost_image(wavelength, folder):
    """
    Python wrapper to run MCFOST for a given wavelength and simulation folder

    Parameters:
    wavelength: wavelength in micrometer
    folder_sim: folder name where the simulation.para file and data_th folder is located

    Returns: 
    None
    """
    if not os.path.exists(folder):
        raise ValueError(f"Folder {folder} does not exist. Please check the folder name and try again.")
    if not os.path.exists(folder+"data_th/sed_rt.fits.gz"):
        raise ValueError(f"SED file does not exist in {folder+'data_th/'}. Please run MCFOST simulation first.")

    
    os.chdir(folder+'/') #this is to change directory to where the simulation.para file is and then run mcfost there
    mcfost.run(folder+'/simulation.para',options = "-img "+str(wavelength), delete_previous=False, silent=True)




def run_mcfost_chi2(param, keys, data_arg, pybads_dir):
    """
    Run MCFOST for given parameters, calculate chi2 for SED and interferometric data
    Parameters
    ----------
    param : list
        List of parameters to set in the simulation.para file.
    keys:
        which parameters are in param. Names correspond to MCFOST parameter file
    data_arg : list
        List containing data for SED and interferometric observations. First element is SED data (tuple of wavelength, flux, error),
        second element is PIONIER data (OIContainer), third element is GRAVITY data (OIContainer),
        fourth element is MATISSE L-band data (OIContainer), fifth element is MATISSE N-band data (OIContainer).
    pybads_dir
        Location where we should run simulation
    Returns
    -------
    chi_total: float
        Total chi2
    chi2_red_total : float
        Total reduced chi2 value for SED and interferometric data.
    loglike: float
        Joint Log-likelihood for optimisation

    """
    #print(param)
    data_sed = data_arg[0]
    container_data_pionier = data_arg[1]
    container_data_gravity = data_arg[2]
    container_data_matisse_l = data_arg[3]
    container_data_matisse_n = data_arg[4]

    pf = ParaFile(pybads_dir+"simulation.para")
    folder_sim=""
    for i in range(0,len(keys)):
        pf.set_param(keys[i], param[i])
        folder_sim+=keys[i]+"_"+str(param[i])+"_"
    
   
     
    simulation_dir = pybads_dir+folder_sim+"/"
    # if not os.path.exists(simulation_dir):
    os.makedirs(simulation_dir, exist_ok=True)  # no error if it already exists

    # Save the modified file
    pf.save(simulation_dir+"simulation.para")
    try:
        #WORKS
        os.chdir(simulation_dir) #this is to change directory to where the simulation.para file is and then run mcfost there
        mcfost.run(simulation_dir+'/simulation.para', delete_previous=True, silent=True)

        for wave in [1.63, 2.20, 3.50, 10.0]:
            mcfost.run(simulation_dir+'/simulation.para',options = "-img "+str(wave), delete_previous=False, silent=True)
    except:
        print(f"MCFOST run failed for parameters: {keys} = {param}")
        return 10e6
    # if os.path.exists(simulation_dir):
    #     print('path exists')
    #     if not os.path.exists(simulation_dir+'data_th/sed_rt.fits.gz'):
    #         pf.save(simulation_dir+"simulation.para")

    #         os.chdir(simulation_dir) #this is to change directory to where the simulation.para file is and then run mcfost there
    #         mcfost.run(simulation_dir+'/simulation.para', delete_previous=True, silent=True)
    #     else:
    #         print('SED file exists')
    #     for wave in [1.63, 2.20, 3.50, 10.0]:
    #         if not os.path.exists(simulation_dir+'data_'+str(wave)+'/'+'RT.fits.gz'):
    #             mcfost.run(simulation_dir+'/simulation.para',options = "-img "+str(wave), delete_previous=False, silent=True)
    #         else:
    #             print('Image at '+str(wave)+' micron exists')

        
    chi2_sed, chi2_reduced_sed, loglike_sed= chi2_SED_with_reddening(folder_sim, pybads_dir, data_wave=data_sed[0], data_flux=data_sed[1],data_err=data_sed[2],
                                       plot=True, description=f"{keys} = {param}")
    
    chi2_pionier, chi2_red_pionier, loglike_pionier, num_points_pionier= monochromatic_chi(simulation_dir, img_dir="data_1.63/", container_data=container_data_pionier, vistype='vis2', plot=True, fig_dir=simulation_dir+'figures/', extra_title="PIONIER 1.63", log_plotv=False)
    chi2_gravity, chi2_red_gravity, loglike_gravity, num_points_gravity= monochromatic_chi(simulation_dir, img_dir="data_2.2/", container_data=container_data_gravity, vistype='vis2', plot=True, fig_dir=simulation_dir+'figures/', extra_title="GRAVITY 2.2", log_plotv=False)
    chi2_matisse_l, chi2_red_matisse_l, loglike_matisse_l, num_points_matisse_l= monochromatic_chi(simulation_dir, img_dir="data_3.5/", container_data=container_data_matisse_l,vistype='vis2', plot=True, fig_dir=simulation_dir+'figures/', extra_title="MATISSE L 3.5", log_plotv=True)
    chi2_matisse_n, chi2_red_matisse_n, loglike_matisse_n, num_points_matisse_n= monochromatic_chi(simulation_dir, img_dir="data_10.0/", container_data=container_data_matisse_n, vistype='vis', plot=True, fig_dir=simulation_dir+'figures/', extra_title="MATISSE N 10.0", log_plotv=False)
    
    
    chi_total= chi2_sed + chi2_pionier + chi2_gravity + chi2_matisse_l + chi2_matisse_n
    num_points_total= len(data_sed[0]) + num_points_pionier + num_points_gravity + num_points_matisse_l + num_points_matisse_n
    
    chi2_red_total = chi_total/(num_points_total-5)
    loglike_total=loglike_sed+loglike_pionier+loglike_gravity+loglike_matisse_l+loglike_matisse_n
   
    return  chi_total, chi2_red_total, loglike_total 