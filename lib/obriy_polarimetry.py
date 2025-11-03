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


import lib.obriy_general as obg
import lib.obriy_interferometry as obi
import lib.obriy_sed as obs
import lib.obriy_mcfost as obm




constants.set_matplotlib_params()  # set project matplotlib parameters


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
    show: bool = False,
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
    roi_half_size: Optional[int] = 50,  # in pixels, for autoscaling window; None = full frame
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
    show: bool = False,
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
        plot_polarimetric_image(Q_conv,   ps, title="Q Conv",     roi_half_size=roi_half_size, image_scale=image_scale, save=save, show=False)
        plot_polarimetric_image(Q_phi_conv, ps, title="Q_phi Conv", roi_half_size=roi_half_size, image_scale=image_scale, save=save, show=False)
        plot_polarimetric_image(I_conv,   ps, title="I Conv",     roi_half_size=roi_half_size, image_scale=image_scale, save=save, show=False)
        plot_polarimetric_image(PI_conv,  ps, title="PI Conv",    roi_half_size=roi_half_size, image_scale=image_scale, save=save, show=False)

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
    if angle==0:
        return image
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
    if max_i> img.shape[0]//2:
        max_i= img.shape[0]//2 -1

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
        plt.close()
    
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

    if r_out_mas> np.min(img.shape)*ps/2.0:
        r_out_mas= np.min(img.shape)*ps/2.0 -1.0
        print(f"Warning: r_out_mas is larger than image size, setting to {r_out_mas} mas")

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
    "r_out_mas": r_out_mas
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
        plt.close()
 


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
    roi_size_half: int = 50,
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
        fov_au=obg.mas2au(fov_mas, distance_pc)
        
        
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
                plot_polarimetric_image(img_q_rescaled, inst_ps_mas, title='Q Rescaled', roi_half_size=roi_size_half, image_scale=image_scale, save=fig_dir, show=False)
                plot_polarimetric_image(q_phi_rescaled, inst_ps_mas, title='Q_phi Rescaled', roi_half_size=roi_size_half, image_scale=image_scale, save=fig_dir, show=False)

                plot_polarimetric_image(img_tot_original, native_ps_mas, title='I tot, original from mcfost', roi_half_size=roi_size_half, image_scale=image_scale, save=fig_dir, show=False)
                plot_polarimetric_image(img_total_rescaled, inst_ps_mas, title='I tot rescaled', roi_half_size=roi_size_half, image_scale=image_scale, save=fig_dir, show=False)
                plot_polarimetric_image(I_conv, inst_ps_mas, title='I tot conv', roi_half_size=roi_size_half, image_scale=image_scale, save=fig_dir, show=False)

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
                                show=False
                                )
                fig.savefig(fig_dir+extra_title+"mcfost_model_comparison.png", dpi=150, bbox_inches='tight')



        # kf.plot_polarimetric_image(q_phi_corr_conv,ps,Q=q_corr_conv,U=u_corr_conv,I=I_conv,title="convolved, unresolved corrected Q phi",bin_factor=(4,4),save=False,snr_threshold=3,noise_level=5e-19,roi_half_size=30,aolp_quiver=True, quiver_scale=0.1)
        # kf.plot_polarimetric_image(pi_rescaled,ps,Q=img_q_rescaled,U=img_u_rescaled,I=img_total_rescaled,title="Q phi",bin_factor=(4,4),save=False,snr_threshold=3,noise_level=2e-17,roi_half_size=30,aolp_quiver=True, quiver_scale=5)
        # kf.plot_polarimetric_image(q_phi_corr_conv, ps, roi_half_size=30, image_scale="asinh")
        # kf.plot_polarimetric_image(Q_conv, ps, roi_half_size=30, image_scale="linear")
        # kf.plot_polarimetric_image(I_conv, ps, roi_half_size=30, image_scale="linear")

        # kf.plot_polarimetric_image(q_phi,pixel_scale,Q=img_q,U=img_u,I=img_tot,title="Q phi",bin_factor=(4,4),image_scale="asinh",save=False,snr_threshold=3,noise_level=1e-17,roi_half_size=100,aolp_quiver=True, quiver_scale=3)



        radial_profile=radial_br_profile(pi_corr_conv, inst_ps_mas,deprojection[0],deprojection[1], R_limit=radial_limit_mas/inst_ps_mas, mode='sum',save=fig_dir+"mcfost_", plot=True,background_annulus_mas=background_annulus_mas)


        az_profile=azimuthal_profile(pi_corr_conv, inst_ps_mas, r_in_mas=azimuthal_r_in_mas, r_out_mas=azimuthal_r_out_mas, plot=True,mode='sum', save=fig_dir+"mcfost_", nbins=azimuthal_nbins, theta0=theta0)

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
    radial_limit_mas: float = 500.0,
    plot: bool = True,
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
    plot_polarimetric_image(obs_data, ps, title='Observed Data check', save=save+'check.png', image_scale='asinh', roi_half_size=30)

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
        prof_obs = radial_br_profile(obs_data, ps, inclination_deg=inc_deg, position_angle_deg=pa_deg, R_limit=radial_limit_mas,
                                     mode=mode, noise_map=obs_err, noise_level=noise_level,xc=xc,yc=yc,
                                     plot=plot, save=save+"obs_")
        R_limit= np.max(prof_obs["i_rad_mas"])

        prof_mod = radial_br_profile(model_data, ps, inclination_deg=inc_deg, position_angle_deg=pa_deg,
                                     R_limit=R_limit, mode=mode, xc=xc,yc=yc,
                                     plot=plot, save=save+"model_")
        
        max_i_rad_mas = min(np.max(prof_obs["i_rad_mas"]), np.max(prof_mod["i_rad_mas"]))
        index_max = np.where(prof_obs["i_rad_mas"] <= max_i_rad_mas)[0][-1]
        
    


        if plot:
            plt.errorbar(prof_obs["i_rad_mas"][:index_max], prof_obs["signal"][:index_max], yerr=prof_obs["error"][:index_max], fmt='o', label='obs')
            plt.errorbar(prof_mod["i_rad_mas"][:index_max], prof_mod["signal"][:index_max], yerr=prof_mod["error"][:index_max], fmt='o', label='model')
            plt.xlabel('Distance from the star (mas)')
            plt.ylabel('Normalised intensity')
            plt.legend()
            plt.savefig(save+'radial_profile_comparison.jpeg',bbox_inches='tight', pad_inches=0.1)
            plt.close()
        
        chi2_sum_radial= ((prof_obs["signal"][:index_max] - prof_mod["signal"][:index_max]) ** 2 / (prof_obs["error"][:index_max] ** 2 + 1e-16)).sum()
        loglike_sum_radial = np.nansum(((prof_obs["signal"][:index_max] - prof_mod["signal"][:index_max]) ** 2)/(prof_obs["error"][:index_max] ** 2 + 1e-16) + np.log(2.0 * np.pi * (prof_obs["error"][:index_max] ** 2 + 1e-16)))
        n_points_radial=len(prof_obs["signal"][:index_max])
    
    if profile_type in ("azimuthal", "both"):
        r_in_mas = 0
        if profile_type=="azimuthal":
            r_out_mas= 500.0
        else:
            r_out_mas =R_limit

        prof_obs_az = azimuthal_profile(obs_data, ps, r_in_mas, r_out_mas,
                                       mode=mode, xc=xc, yc=yc, nbins=az_nbins,
                                       plot=plot, save=save+"obs_")
        
        prof_mod_az = azimuthal_profile(model_data, ps, r_in_mas, r_out_mas,
                                       mode=mode, xc=xc, yc=yc, nbins=az_nbins,
                                       plot=plot, save=save+"model_")
        if plot:
            plt.plot(prof_obs_az["theta_deg_centers"], prof_obs_az["value"], 'o', label='obs')
            plt.plot(prof_mod_az["theta_deg_centers"], prof_mod_az["value"], 'o', label='model')
            plt.xlabel('Position angle (deg)')
            plt.ylabel('Normalised intensity')
            plt.legend()
            plt.savefig(save+'azimuthal_profile_comparison.jpeg',bbox_inches='tight', pad_inches=0.1)
            plt.close()
        
        chi2_sum_az = ((prof_obs_az["value"] - prof_mod_az["value"]) ** 2 / (prof_obs_az["std"] ** 2 + 1e-16)).sum() #this is weighted least-squares χ²
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
    print(f"Chi2: {chi2_sum}, Reduced Chi2: {chi2_red}, Log-Likelihood: {loglike}, Data points: {n_data_points}")
    
    return chi2_sum, chi2_red, loglike, n_data_points




