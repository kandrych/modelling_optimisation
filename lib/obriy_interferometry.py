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

import lib.obriy_general as obg
import lib.obriy_sed as obs
import lib.obriy_mcfost as obm
import lib.obriy_polarimetry as obp



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
    chi2_sum: float
        Total sum of chi2.
    chi2_red : float
        Reduced chi2 value.
    loglike: float
        Log-likelihood.
    n_data_points: int
        Number of data points. 
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
    objective = obg.pick_output(chi2_for_optimisation_overresolved, idx=2, cast=float)

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
