import copy
import distroi
import numpy as np
import pandas as pd
from typing import Literal, Tuple, Dict, Optional, Union, Any, List
from pathlib import Path
from collections.abc import Sequence

import distroi
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
from scipy.stats import chisquare
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
import lib.obriy_sed as obs
import lib.obriy_mcfost as obm
import lib.obriy_polarimetry as obp


plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.edgecolor"] = "grey"
plt.rcParams["legend.framealpha"] = 0.5
plt.rcParams["lines.markersize"] = 6.0
plt.rcParams["lines.linewidth"] = 2.0

plt.rc("font", size=16)  # controls default text sizes
plt.rc("axes", titlesize=14)  # fontsize of the axes title
plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
plt.rc("ytick", labelsize=14)  # fontsize of the tick labels
plt.rc("legend", fontsize=14)  # legend fontsize
plt.rc("figure", titlesize=14)  # fontsize of the figure title



#######################################
# INTERFEROMETRY
#######################################


def validate_interferometric_data(container, vistype):
    """Validate observations once; preserve the existing positive-error selection."""
    field = {"vis2": "v2", "vis": "v"}.get(vistype)
    if field is None:
        raise ValueError("vistype must be 'vis2' or 'vis'.")
    data = np.asarray(getattr(container, field))
    errors = np.asarray(getattr(container, field + "_err"))
    if data.ndim != 1 or errors.shape != data.shape:
        raise ValueError("Interferometric values and errors must be matching 1D arrays.")
    if not np.all(np.isfinite(errors)):
        raise ValueError("Non-finite interferometric observational errors.")
    valid = errors > 0
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two positive-error interferometric samples are required for chi2/(N-1).")
    if not np.all(np.isfinite(data[valid])):
        raise ValueError("Non-finite retained interferometric observations.")
    return valid


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
   
    valid = validate_interferometric_data(container_data, vistype)
    if not np.all(np.isfinite(vismod[valid])):
        raise ValueError("Non-finite model predictions at retained interferometric samples.")
    if sigma_sys_frac is not None and (not np.isfinite(sigma_sys_frac) or sigma_sys_frac < 0):
        raise ValueError("Systematic fractional error must be finite and non-negative.")

    # Loop over the observation-selected samples; never mask a bad model prediction.
    for i in range(len(visdata)):
        if valid[i]:
            if sigma_sys_frac is not None:
                varience = viserrdata[i] ** 2 + (sigma_sys_frac * vismod[i])**2
            else:
                varience = viserrdata[i] ** 2
            chi2_sum += ((visdata[i] - vismod[i]) ** 2) / varience
            loglike_sum+=((visdata[i] - vismod[i]) ** 2)/varience +np.log(2.0 * np.pi * varience)
            n_data_points += 1  # Count only points with valid error bars

    if not np.isfinite(chi2_sum) or not np.isfinite(loglike_sum):
        raise ValueError("Non-finite interferometric score; check fluxes and uncertainties.")
    chi2_red = chi2_sum / (n_data_points-1)
    #chi2_red_scipy=chisquare(visdata, f_exp=vismod, ddof=1, sum_check=False)[0]
    #print(f"Chi2_red calculation check: custom={chi2_red}, scipy={chi2_red_scipy}")
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
    extra_title: str = None,
    comparison_model: OIContainer = None
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
        label="model" if comparison_model is None else "MCFOST + secondary",
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

    if comparison_model is not None:
        comparison_values = comparison_model.v2 if plot_vistype == "vis2" else comparison_model.v
        ax[0].scatter(basedata, comparison_values, label="MCFOST alone",
                      marker="o", facecolor="white", edgecolor="k", s=4, alpha=0.6)
        ax[1].scatter(basedata, (comparison_values - visdata) / viserrdata,
                      marker="o", facecolor="white", edgecolor="k", s=4, alpha=0.6)
        # Include both predictions when setting the plot limits below.
        vismod = np.maximum(vismod, comparison_values)

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





def _float64_wavelengths(container):
    """Prevent float32 overflow in Distroi's blackbody frequency cubing."""
    result = copy.copy(container)
    for field in ("v_wave", "v2_wave", "t3_wave"):
        values = getattr(container, field, None)
        if values is not None:
            setattr(result, field, np.asarray(values, dtype=np.float64))
    return result


def plot_secondary_comparison(container_data, img_ffts, img_sed, fig_dir,
                              vistype, extra_title, log_plotv, ebminv, reddening_law):
    """Additional comparison using the existing visibility plotting style."""
    from lib.obriy_sed import redden_flux
    data = _float64_wavelengths(container_data)
    baseline = distroi.oi_container_calc_image_fft_observables(data, img_ffts, img_sed=img_sed)
    if baseline.vis_in_fcorr and ebminv != 0:
        law = reddening_law or str(Path(__file__).resolve().parent.parent
                                  / "utils" / "ISMreddening_law_Cardelli1989.dat")
        baseline.v = redden_flux(baseline.v_wave, baseline.v, law, ebminv)
    secondary = calc_observables_with_secondary(
        data, img_ffts, img_sed, ebminv=ebminv, reddening_law=reddening_law)
    _, reduced, _, _ = oi_container_chi2(data, secondary, vistype=vistype)
    output = str(Path(fig_dir) / "secondary_comparison") if fig_dir is not None else None
    oi_container_plot_data_vs_model(
        data, secondary, fig_dir=output, log_plotv=log_plotv,
        plot_vistype=vistype, show_plots=False, chi_plot=reduced,
        extra_title=extra_title, comparison_model=baseline)
    return {"MCFOST only": baseline, "+ secondary": secondary}


def calc_observables_with_secondary(container_data, img_ffts, img_sed,
                                    background_fraction=None, background_wavelength=None,
                                    ebminv=0.0, reddening_law=None):
    """Add the accretion disc with the same intrinsic normalisation as the SED.

    Its 3.9% fraction at 1.65 micron excludes optional overresolved emission.
    Distroi requires component fractions of the final total at a shared reference;
    convert those fractions without changing the absolute accretion-disc flux.
    """
    from lib.obriy_sed import add_blackbody_component, redden_flux

    if not np.isfinite(ebminv) or ebminv < 0:
        raise ValueError("Foreground E(B-V) must be finite and non-negative.")

    if img_sed is None:
        raise ValueError("The unmodified MCFOST SED is required to normalise the secondary.")
    reference = 1.65 if background_fraction is None else background_wavelength

    if reference is None or not np.isfinite(reference) or reference <= 0:
        raise ValueError("A positive background reference wavelength is required.")

    waves = np.asarray(img_sed.wavelengths)
    if not waves.min() <= 1.65 <= waves.max():
        raise ValueError("The MCFOST SED must cover 1.65 micron.")

    # Reuse the photometric lambda*F_lambda interpolation and blackbody exactly.
    order = np.argsort(waves)
    lam_flam = waves * np.asarray(img_sed.flam)
    query = np.unique(np.append(waves, reference))
    base = np.interp(query, waves[order], lam_flam[order])
    _, component = add_blackbody_component(query, base)
    frequency = constants.SPEED_OF_LIGHT / (reference * constants.MICRON2M)
    secondary_jy = np.interp(reference, query, component) * 1e23 / frequency
    base_jy = float(img_sed.get_flux(x=frequency, flux_form="fnu"))

    if not np.isfinite(base_jy) or base_jy <= 0:
        raise ValueError("The reference MCFOST flux must be finite and positive.")
    secondary = distroi.PointSource(
        coords=(0.0, 0.0),
        sp_dep=distroi.BlackBodySpecDep(temp=4000.0))
    components = [secondary]
    fractions = [secondary_jy / (base_jy + secondary_jy)]

    if background_fraction is not None:
        fraction = float(background_fraction)
        if not np.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError("Background fraction must be finite and in [0, 1).")
        # B/(M+S+B)=fraction; S remains fixed as B varies.
        components.insert(0, distroi.Overresolved(sp_dep=distroi.FlatSpecDep(flux_form="flam")))
        fractions = [fraction, (1 - fraction) * fractions[0]]

    resulting=distroi.oi_container_calc_image_fft_observables(
        _float64_wavelengths(container_data), img_ffts, img_sed=img_sed, geom_comps=components,
        geom_comp_flux_fracs=fractions, ref_wavelength=reference)

    # Apply the common foreground screen after combining all intrinsic components.
    # It cancels in normalised visibility, but attenuates correlated flux in Jy.
    if resulting.vis_in_fcorr and ebminv != 0:
        if reddening_law is None:
            reddening_law = str(Path(__file__).resolve().parent.parent
                                / "utils" / "ISMreddening_law_Cardelli1989.dat")
        resulting.v = redden_flux(
            resulting.v_wave, resulting.v, reddening_law, ebminv)
    return resulting


def chi2_for_optimisation_overresolved(frac, ref_wavelength, container_data, img_ffts, img_sed=None,
                                      vistype="vis2", ebminv=0.0, reddening_law=None) -> Tuple[float, float, float, int]:
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
    container_model = calc_observables_with_secondary(
        container_data, img_ffts, img_sed, float(frac), ref_wavelength,
        ebminv=ebminv, reddening_law=reddening_law)

    chi2, chi2_red, loglike, n_data=oi_container_chi2(container_data, container_model, vistype=vistype)
    return chi2, chi2_red, loglike, n_data

def monochromatic_chi(
        simulation_dir: str,
        img_dir: str| Sequence[str],
        container_data: OIContainer,
        vistype: str='vis2',
        plot: bool=False,
        fig_dir: str=None,
        extra_title: str=None,
        log_plotv: bool=False,
        ebminv: float=0.0,
        reddening_law: str=None
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
  
    img_ffts=distroi.read_image_list(simulation_dir, img_dir)
    img_sed = distroi.read_sed_mcfost(str(Path(simulation_dir) / "data_th" / "sed_rt.fits.gz"))
    container_model = calc_observables_with_secondary(container_data, img_ffts, img_sed, ebminv=ebminv, reddening_law=reddening_law)
    chi2, chi2_red, likelihood, num_points=oi_container_chi2(container_data, container_model, vistype=vistype)

    if plot:    
        with obg.diagnostic_plot("Monochromatic interferometry data/model comparison"):
            oi_container_plot_data_vs_model(
                container_data,
                container_model,
                fig_dir=fig_dir,
                log_plotv=log_plotv,
                plot_vistype=vistype,
                show_plots=False,
                chi_plot=chi2_red,
                extra_title=extra_title)

    if plot:
        with obg.diagnostic_plot("Monochromatic interferometry secondary comparison"):
            plot_secondary_comparison(
                container_data, img_ffts, img_sed, fig_dir, vistype,
                extra_title, log_plotv, ebminv, reddening_law)

    return chi2, chi2_red, likelihood, num_points


def background_objective(frac: float, ref_wavelength: float, container_data: Any, img_ffts: Any, img_sed=None,
                         vistype="vis2", ebminv=0.0, reddening_law=None) -> float:
    _, chi2_red, _, _ = chi2_for_optimisation_overresolved(
        frac=frac,
        ref_wavelength=ref_wavelength,
        container_data=container_data,
        img_ffts=img_ffts,
        img_sed=img_sed,
        vistype=vistype, ebminv=ebminv, reddening_law=reddening_law,
    )
    return float(chi2_red)

def monochromatic_chi_with_background(
        simulation_dir: str,
        img_dir: str,
        container_data: OIContainer,
        img_sed: Any = None,
        wave_for_background: float = None,
        frac_for_background: float=None,
        vistype: str='vis2',
        plot: bool=False,
        fig_dir: str=None,
        extra_title: str=None,
        log_plotv: bool=False,
        ebminv: float=0.0,
        reddening_law: str=None
) -> Tuple[float, float, float, int, float]:
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
    img_sed : distroi.data.sed.SED, optional
        SED object containing the model SED to scale for the overresolved flux. If None, load the unmodified MCFOST SED from the simulation directory.
    wave_for_background : float
        Wavelength in micrometer for which the background is calculated.
    frac_for_background : float, optional
        Fraction of the background (overresolved) flux. Default is None and means that the fraction is to be optimized.
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
    frac_best : float
        Best-fit background flux fraction in the model. If frac_for_background is provided, this will be equal to that value.
    """
    
    
    img_ffts = distroi.read_image_list(simulation_dir, img_dir)
    if img_sed is None:
        img_sed = distroi.read_sed_mcfost(str(Path(simulation_dir) / "data_th" / "sed_rt.fits.gz"))

    if frac_for_background is None:
        frac_min = minimize_scalar(
            lambda x: background_objective(x, ref_wavelength=wave_for_background, container_data=container_data, img_ffts=img_ffts, img_sed=img_sed, vistype=vistype, ebminv=ebminv, reddening_law=reddening_law),
            bounds=(0.0, 0.5),
            method="bounded",
            options={"xatol": 1e-4}
        )
        frac_best = float(frac_min.x)
    else:
        frac_best = frac_for_background
    
    container_model = calc_observables_with_secondary(
        container_data, img_ffts, img_sed, frac_best, wave_for_background,
            ebminv=ebminv, reddening_law=reddening_law)

    chi2, chi2_red,loglike, num_points=oi_container_chi2(container_data, container_model, vistype=vistype)

    if plot:
        with obg.diagnostic_plot("Monochromatic interferometry with background comparison"):
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

    if plot:
        with obg.diagnostic_plot("Monochromatic interferometry secondary comparison with background enabled"):
            plot_secondary_comparison(
                container_data, img_ffts, img_sed, fig_dir, vistype,
                extra_title, log_plotv, ebminv, reddening_law)

    return chi2, chi2_red, loglike, num_points, frac_best





def chromatic_chi(
        simulation_dir: str,
        img_dir: str | list[str],
        container_data: OIContainer,
        vistype: str='vis2',
        img_sed: sed.SED=None,
        wave_for_background: float=None,
        frac_for_background: float=None,
        plot: bool=False,
        fig_dir: str=None,
        extra_title: str=None,
        log_plotv: bool=False,
        ebminv: float = 0.0,
        reddening_law: str = None
) -> Tuple[float, float,float, int]:
    """
    Wrapper to calculate chi2 and reduced chi2 for a chromatic model without background.

    ----------
    Parameters
    
    simulation_dir : str
        Directory where the MCFOST simulation is located.
    img_dir : str | list[str]
        Directory or list of directories where the MCFOST images for specific wavelengths are located.
    container_data : OIContainer
        Container with data observables.
    vistype : {'vis2', 'vis', 'fcorr'}, optional
        Type of visibility to be used in the chi2 calculation. Default is 'vis2'.
    img_sed : distroi.data.sed.SED, optional
        SED object containing the model SED to scale for the overresolved flux. If None, load the unmodified MCFOST SED from the simulation directory.
    wave_for_background : float, optional
        Wavelength in micrometer for which the background is calculated. Default is None and means that there is no background (overresolved) flux.
    frac_for_background : float, optional
        Fraction of the background (overresolved) flux. Default is None and means that the fraction is to be optimized.
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

    if img_sed is None:
        img_sed = distroi.read_sed_mcfost(str(Path(simulation_dir) / "data_th" / "sed_rt.fits.gz"))
    img_dir = [img_dir] if isinstance(img_dir, str) else list(img_dir)
    img_ffts=[]
    wavelengths=[]

    for directory in img_dir:
        #wavelength_img= directory.split('_')[-1]
        #wavelength_img= float(wavelength_img.replace('/',''))
        #print(f"[obriy_interferometry, chromatic_chi] Reading image for wavelength {wavelength_img} from {simulation_dir}/{directory}")
        img_file_paths = sorted(glob.glob(f"{simulation_dir}/{directory}/**/*RT.fits.gz", recursive=True))
        for img_path in img_file_paths:
            img= distroi.read_image_mcfost(img_path)
            img_ffts.append(img)  # append to the list of Image objects
            wavelengths.append(img.wavelength)  # append wavelength
       
    
    wavelengths, img_ffts = list(zip(*sorted(zip(wavelengths, img_ffts))))  # sort the objects in wavelength



    if wave_for_background is not None:

        if frac_for_background is None:
            frac_min = minimize_scalar(
                lambda x: background_objective(x, ref_wavelength=wave_for_background, container_data=container_data, img_ffts=img_ffts, img_sed=img_sed, vistype=vistype, ebminv=ebminv, reddening_law=reddening_law),
                bounds=(0.0, 0.5),
                method="bounded",
                options={"xatol": 1e-4}
            )
            frac_best = float(frac_min.x)
        else:
            frac_best = frac_for_background
        
        container_model = calc_observables_with_secondary(
            container_data, img_ffts, img_sed, frac_best, wave_for_background,
            ebminv=ebminv, reddening_law=reddening_law)
    else:
        # No background component, just the accretion secondary that is fixed for IRAS08 based on Hillen et al 2016. For other objects, this should be changed to a more appropriate value or made a free parameter in the optimisation.
        container_model = calc_observables_with_secondary(container_data, img_ffts, img_sed, ebminv=ebminv, reddening_law=reddening_law)
      
    
    chi2, chi2_red, likelihood, num_points=oi_container_chi2(container_data, container_model, vistype=vistype)

    if plot:    
        with obg.diagnostic_plot("Chromatic interferometry data/model comparison"):
            oi_container_plot_data_vs_model(
                container_data,
                container_model,
                fig_dir=fig_dir,
                log_plotv=log_plotv,
                plot_vistype=vistype,
                show_plots=False,
                chi_plot=chi2_red,
                extra_title=extra_title)

    if plot:
        with obg.diagnostic_plot("Chromatic interferometry secondary comparison"):
            plot_secondary_comparison(
                container_data, img_ffts, img_sed, fig_dir, vistype,
                extra_title, log_plotv, ebminv, reddening_law)

    return chi2, chi2_red, likelihood, num_points

def distroi_redden_copy(
        img: image.Image,
        ebminv: float,
        reddening_law_path: str = None,
    ) -> image.Image:
        """Redden the image.

        Further reddens the model image according to the appropriate E(B-V) and a corresponding reddening law.

        Parameters
        ----------
        ebminv : float
            E(B-V) reddening factor to be applied.
        reddening_law : str, optional
            Path to the reddening law to be used. Defaults to the ISM reddening law by Cardelli (1989) in DISTROI's
            'utils/ISM_reddening folder'. See this file for the expected formatting of your own reddening laws.

        Returns
        -------
        None
        """
        if reddening_law_path is None:
            folder_of_script = Path(__file__).resolve().parent
             # one above the folder of the script:
            reddening_law_path = str(folder_of_script.parent / "utils"/"ISMreddening_law_Cardelli1989.dat")

        img.img = constants.redden_flux(
            img.wavelength,
            img.img,  # apply additional reddening to the image
            ebminv,
            reddening_law=reddening_law_path,
        )
        if img.fft is not None:
            img.fft = constants.redden_flux(
                img.wavelength, img.fft, ebminv, reddening_law_path
            )  # apply additional reddening to the fft
        img.ftot = constants.redden_flux(
            img.wavelength, img.ftot, ebminv, reddening_law_path
        )  # apply additional reddening to the toal flux
        return img