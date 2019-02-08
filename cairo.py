"""Code for planning a space-based water-band mission
"""

from __future__ import division, print_function

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict
from scipy.interpolate import interp1d
from scipy.integrate import simps
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------------------------------------------------------
# Loading Profiles, Calculating Magnitudes
# -----------------------------------------------------------------------------
def load_and_extend_profiles(filt="j", min_wl=1200, max_wl=210000):
    """Load in the specified filter profile and zero pad both ends in 
    preparation for feeding into an interpolation function.
    """
    filters_2mass = np.array(["j", "h", "k"])
    filters_gaia = np.array(["g", "bp", "rp"])
    
    # 2MASS filter profiles
    if filt in filters_2mass:
        # Load the filter profile, and convert to Angstroms
        filt_profile = np.loadtxt("2mass_%s_profile.txt" % filt)
        filt_profile[:,0] = filt_profile[:,0] * 10**4
    
    # Gaia filter profiles
    elif filt in filters_gaia:
        # Import gaia profiles, and set undefined values from 99.99 to 0
        gaia_profiles = np.loadtxt("GaiaDR2_RevisedPassbands.dat")
        gaia_profiles[gaia_profiles==99.99] = 0
        
        # Format of file is [wl, G, G_err, Bp, Bp_err, Rp, Rp_err]
        wl = gaia_profiles[:,0]
        
        col = int(np.argwhere(filters_gaia==filt)) * 2 + 1
        
        passband = gaia_profiles[:,col]
        
        filt_profile = np.vstack([wl, passband]).T
        
    else:
        raise Exception("Invalid filter profile")
    
    # Pre- and post- append zeros from min_wl to max_wl
    prepad_wl = np.arange(min_wl, filt_profile[0,0], 1000)
    prepad = np.zeros((len(prepad_wl), 2))
    prepad[:, 0] = prepad_wl
    
    postpad_wl = np.arange(filt_profile[-1,0], max_wl, 10000)
    postpad = np.zeros((len(postpad_wl), 2))
    postpad[:, 0] = postpad_wl
    
    filt_profile = np.concatenate((prepad, filt_profile, postpad))
    
    return filt_profile


def extend_filter_profile(filt_profile, min_wl=1200, max_wl=210000):
    """Zero pad both ends in preparation for interpolation function.
    """
    # Pre- and post- append zeros from min_wl to max_wl
    prepad_wl = np.arange(min_wl, filt_profile[0,0], 1000)
    prepad = np.zeros((len(prepad_wl), 2))
    prepad[:, 0] = prepad_wl
    
    postpad_wl = np.arange(filt_profile[-1,0], max_wl, 10000)
    postpad = np.zeros((len(postpad_wl), 2))
    postpad[:, 0] = postpad_wl
    
    filt_profile = np.concatenate((prepad, filt_profile, postpad))
    
    return filt_profile
     

def calculate_band_flux(wavelengths, fluxes, template_profile, filt):
    """Function to put a given filter profile on a different wavelength scale, 
    and compute the fluxes given a spectrum. 
    """
    # Set up the interpolator for the filter profile
    calc_filt_profile = interp1d(template_profile[:,0], template_profile[:,1],
                                 kind="linear")
                                 
    # Calculate the profile for the given wavelength scale
    filt_profile = calc_filt_profile(wavelengths)
    
    # planck constant
    h = 6.62607015 * 10**-34 # J sec
    c = 299792458 * 10 **10
    
    # Assume solar radii for synthetic fluxes/star, and scale to 10 pc distance
    r_sun = 6.95700*10**8        # metres
    d_10pc = 10 * 3.0857*10**16  # metres
    
    flux = (r_sun / d_10pc)**2 * fluxes
    
    # Compute the flux density of the star
    mean_flux_density = (simps(flux*filt_profile*wavelengths, wavelengths)
                         / simps(filt_profile*wavelengths, wavelengths))
    
    # Define Vega fluxes (erg s^-1 cm^-2 A^-1) and zeropoints for each filter
    # band. For the W band, treat it as the average of J and K.
    # Vega fluxes from Casagrande & VendenBerg 2014, Table 1
    # Band zeropoints from Bessell, Castelli, & Plez 1998, Table A2
    vega_mags = {"B":[6.3170* 10**-9, -0.120],
                 "V":[3.6186* 10**-9, 0],
                 "Rc":[2.1652* 10**-9, 0.186],
                 "Ic":[1.1327* 10**-9, 0.444],
                 "J":[3.129* 10**-10, 0.899], 
                 "H":[1.113* 10**-10, 1.379], 
                 "Ks":[4.283* 10**-11, 1.886]}  # erg cm-2 s-1 A-1
    vega_mags["W"] = [np.mean([vega_mags["J"][0], vega_mags["H"][0]]),
                      np.mean([vega_mags["J"][1], vega_mags["H"][1]])]
    
    # Calculate the magnitude of each star w.r.t. Vega
    # mag_star = -2.5 log10(F_star / F_vega) + zero-point
    mag = -2.5 * np.log10(mean_flux_density/vega_mags[filt][0]) + vega_mags[filt][1]
    
    return mag
    

def calculate_band_flux_vega(wavelengths, fluxes):
    """Function to put a given filter profile on a different wavelength scale, 
    and compute the fluxes given a spectrum. 
    """
    # Constants
    filters = ["B", "V", "Rc", "Ic", "J", "W", "H", "Ks"]
    
    filter_profile_files = {"B":"filters/Bessel_B-1.txt",
                            "V":"filters/Bessel_V-1.txt",
                            "Rc":"filters/Bessel_R-1.txt",
                            "Ic":"filters/Bessel_I-1.txt",
                            "J":"filters/2mass_j_profile.txt",
                            "W":"filters/emu_w.txt",
                            "H":"filters/2mass_h_profile.txt",
                            "Ks":"filters/2mass_k_profile.txt"}
    
    # Put the 2MASS wavelength scale in Angstroms from um
    wl_scale_factor = {"B":10, "V":10, "Rc":10, "Ic":10, "J":10**4, "W":1, "H":10**4, "Ks":10**4}
    
    vega_flux = {"B":6.3170* 10**-9,
                 "V":3.6186* 10**-9,
                 "Rc":2.1652* 10**-9,
                 "Ic":1.1327* 10**-9,
                 "J":3.129* 10**-10, 
                 "W":2.121* 10**-10,    # Mean of J and H values
                 "H":1.113* 10**-10, 
                 "Ks":4.283* 10**-11} # erg cm^-2 s^-1 A^-1
    
    vega_zp = {"B":-0.120,
               "V":0,
               "Rc":0.186,
               "Ic":0.444,
               "J":0.899, 
               "W":1.139,   # Mean of J and H values
               "H":1.379, 
               "Ks":1.886}
               
    # Assume solar radii for synthetic fluxes/star, and scale to 10 pc distance
    r_sun = 6.95700*10**8        # metres
    d_10pc = 10 * 3.0857*10**16  # metres
    
    vega_mags = OrderedDict()
    
    # Compute all fluxes at once for simplicity
    for filt in filters:
        # Load the filter profile
        filt_profile = np.loadtxt(filter_profile_files[filt])
        
        # Scale the profile wavelengths
        filt_profile[:,0] = wl_scale_factor[filt] * filt_profile[:,0]
        
        # Extend the filter profile to the limits of the wavelength scale
        filt_profile = extend_filter_profile(filt_profile)
        
        # Put the filter profile on the same wavelength scale
        calc_filt_profile = interp1d(filt_profile[:,0], filt_profile[:,1],
                                     kind="linear")
        filt_profile = calc_filt_profile(wavelengths)
                                     
        # Scale the fluxes by stellar size and interstellar distance
        flux = (r_sun / d_10pc)**2 * fluxes

        # Compute the flux density of the star
        f_lambda = (simps(flux*filt_profile*wavelengths, wavelengths)
                    / simps(filt_profile*wavelengths, wavelengths))
        
        mag = -2.5 * np.log10(f_lambda/vega_flux[filt]) + vega_zp[filt]
                
        # Add to the dictionary of returned fluxes and move on
        vega_mags[filt] = mag
        
    return vega_mags
    


def calculate_band_flux_ab(wavelengths, fluxes):
    """
    """
    # Constants
    f0_lambda = 3.631 * 10 **-9 # ergs s^-1 cm^-2 A^-1
    zp_lambda = -21.10
    f0_nu = 3.631 * 10 **-20 # ergs s^-1 cm^-2 Hz^-1
    zp_nu = -48.6
    
    filters = ["u", "g", "r", "i", "z", "Y"]
    
    # AB Magnitude Zeropoints from:
    # http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=SLOAN/
    # SDSS.u&&mode=browse&gname=SLOAN&gname2=SDSS#filter
    ab_zp = {"u":8.423e-9,
             "g":5.055e-9,
             "r":2.904e-9,
             "i":1.967e-9,
             "z":1.375e-9,
             "Y":1.018e-9} # erg cm^-2 s^-1 A^-1
    
    filter_profile_files = {"u":"filters/SLOAN_SDSS.u.dat",
                            "g":"filters/SLOAN_SDSS.g.dat",
                            "r":"filters/SLOAN_SDSS.r.dat",
                            "i":"filters/SLOAN_SDSS.i.dat",
                            "z":"filters/SLOAN_SDSS.z.dat",
                            "Y":"filters/gemini-Y.txt"}
    
    # Assume solar radii for synthetic fluxes/star, and scale to 10 pc distance
    r_sun = 6.95700*10**8        # metres
    d_10pc = 10 * 3.0857*10**16  # metres         
             
    ab_mags = OrderedDict()

    # Compute all fluxes at once for simplicity
    for filt in filters:
        # Load the filter profile
        filt_profile = np.loadtxt(filter_profile_files[filt])
        
        # Extend the filter profile to the limits of the wavelength scale
        filt_profile = extend_filter_profile(filt_profile)
        
        # Put the filter profile on the same wavelength scale
        calc_filt_profile = interp1d(filt_profile[:,0], filt_profile[:,1],
                                     kind="linear")
        filt_profile = calc_filt_profile(wavelengths)
                                     
        # Scale the fluxes by stellar size and interstellar distance
        flux = (r_sun / d_10pc)**2 * fluxes

        # Compute the flux density of the star
        # Equation 6 from Casagrande & VendenBerg 2014
        mag = (-2.5*np.log10(simps(wavelengths*flux*filt_profile, wavelengths)
                             / (f0_lambda*simps(wavelengths*filt_profile)))
               -2.5*np.log10(simps(wavelengths*filt_profile, wavelengths)
                              / simps(filt_profile/wavelengths))
               + 18.6921)
                
        # Add to the dictionary of returned fluxes and move on
        ab_mags[filt] = mag
        
    return ab_mags
        

# -----------------------------------------------------------------------------
# Plotting Filter Profiles
# -----------------------------------------------------------------------------
def plot_jhk_profiles():
    """Plot the JHK filter profiles.
    """
    profiles = ["2mass_j_profile.txt", "2mass_h_profile.txt", 
                "2mass_k_profile.txt"]
    filters = ["J", "H", "K"]
    colours = ["red", "firebrick", "maroon"]
    
    for filt_i, filt in enumerate(filters):
        # Load in the profile, convert to Angstroms, and scale y
        filt_profile = np.loadtxt(profiles[filt_i])
        filt_profile[:,0] = filt_profile[:,0] * 10**4
        filt_profile[:,1] = filt_profile[:,1]# * plt.gca().get_ybound()[1]
        
        plt.fill_between(filt_profile[:,0], filt_profile[:,1], label=filt,
                         alpha=0.25, color=colours[filt_i])
                         

def plot_gaia_profiles(in_angstroms=True):
    """Plot the Gaia G, Bp, and Rp filter profiles.
    """
    # Import gaia profiles, and set undefined values from 99.99 to 0
    gaia_profiles = np.loadtxt("GaiaDR2_RevisedPassbands.dat")
    gaia_profiles[gaia_profiles==99.99] = 0
    
    if in_angstroms:
        gaia_profiles[:,0] *= 10

    filters = ["G", "Bp", "Rp"]
    colours = ["green", "blue", "red"]
    
    for filt_i, filt in enumerate(filters):
        plt.fill_between(gaia_profiles[:,0], gaia_profiles[:,filt_i*2+1], 
                         label=filt, alpha=0.25, color=colours[filt_i])


def plot_wise_profiles(in_angstroms=True):
    """Plot the WISE W1, W2, W3, and W4 filter profiles
    """
    # Import gaia profiles, and set undefined values from 99.99 to 0
    gaia_profiles = np.loadtxt("GaiaDR2_RevisedPassbands.dat")
    gaia_profiles[gaia_profiles==99.99] = 0
    
    if in_angstroms:
        scale = 10**4
    else:
        scale = 10**3

    filters = ["W1", "W2", "W3", "W4"]
    #colours = ["green", "blue", "red"]
    
    for filt_i, filt in enumerate(filters):
        filt_profile = np.loadtxt("RSR-W%s.EE.txt" % (filt_i+1))
        plt.fill_between(filt_profile[:,0]*scale, filt_profile[:,1], 
                         label=filt, alpha=0.25)#, color=colours[filt_i])


# -----------------------------------------------------------------------------
# Importing Model Grids
# -----------------------------------------------------------------------------
def import_marcs_flx_grid(n_wl, import_full_grid=False):
    """Import the MARCS model grid of flx files per the local directory 
    structure.
    """
    # Define the available grid points
    if import_full_grid:
        temps = np.arange(2500, 4100, 100)
        fehs = np.array([-5, -4, -3, -2.5, -2, -1.5, -1, -0.75, -0.5, -0.25, 0, 
                         0.25, 0.5, 0.75, 1])
    else:
        temps = np.arange(2500, 4100, 200)
        fehs = np.arange(-5, 2, 1)
    
    loggs = np.arange(3, 6, 0.5)
    
    grid = np.zeros((len(temps), len(loggs), len(fehs), n_wl))
    
    grid_files = glob.glob("/Users/adamrains/MARCS/flx/p*g*z*.flx")
    grid_files.sort()
    
    # Load files into grid, taking a little over half of the wavelength info
    ratio = 0.6
    
    for gfile in grid_files:
        # Determine the parameters of the file
        gfn = gfile.split("/")[-1]
        
        try:
            temp_i = np.where(temps==int(gfn.split("_")[0][1:]))[0][0]
            logg_i = np.where(loggs==float(gfn.split("_")[1][1:]))[0][0]
            feh_i = np.where(fehs==float(gfn.split("_")[5][1:]))[0][0]
        except:
            continue
        fluxes = np.loadtxt(gfile)
        
        grid[temp_i, logg_i, feh_i, :] = fluxes[:n_wl]
        
    return grid


def import_marcs_flx_grid_server(n_wl, import_full_grid=False):
    """Import the MARCS model grid of flx files per the server directory 
    structure.
    """
    if import_full_grid:
        temps = np.arange(2500, 4200, 100)
        fehs = np.arange(-2, 1, 0.25)
    else:
        temps = np.arange(2500, 4200, 200)
        fehs = np.arange(-2, 1, 0.5)
        
    loggs = np.arange(-0.5, 5.5, 0.5)
    
    grid = np.zeros((len(temps), len(loggs), len(fehs), n_wl))
    
    # Folders sorted by logg, two different kinds of models: spherical and 
    # plane parallel
    grid_files = glob.glob("/priv/mulga1/luca/MARCS/ppl_*/*.flx.gz")
    grid_files += glob.glob("/priv/mulga1/luca/MARCS/sph_*/*.flx.gz")
    grid_files.sort()
    
    # Load files into grid, taking a little over half of the wavelength info
    ratio = 0.6
    
    for gfile in grid_files:
        # Determine the parameters of the file
        gfn = gfile.split("/")[-1]
        
        try:
            temp_i = np.where(temps==int(gfn.split(":")[0][1:]))[0][0]
            logg_i = np.where(loggs==float(gfn.split(":")[1][1:]))[0][0]
            feh_i = np.where(fehs==float(gfn.split(":")[5][1:]))[0][0]
        except:
            continue
        fluxes = np.loadtxt(gfile)
        
        grid[temp_i, logg_i, feh_i, :] = fluxes[:n_wl]
        
    return grid, temps, fehs, loggs


def import_marcs_flx_grid_alpha(n_wl=60434):
    """Import the MARCS model grid of flx files per the server directory 
    structure.
    """
    temps = np.arange(2500, 4500, 100)
    fehs = np.arange(-2, 0.25, 0.25)  
    #loggs = np.array([4.5])
    loggs = np.array([4.5, 5.0, 5.5])
    
    a04_grid = np.zeros((len(temps), len(loggs), len(fehs), n_wl))
    aNE_grid = np.zeros((len(temps), len(loggs), len(fehs), n_wl))
    aSTD_grid = np.zeros((len(temps), len(loggs), len(fehs), n_wl))
    
    # Folders sorted by logg, two different kinds of models: spherical and 
    # plane parallel
    a04_grid_files = glob.glob("marcs/a04_ppl_vtur2_log*/*.flx.gz")
    aNE_grid_files = glob.glob("marcs/aNE_ppl_vtur2_log*/*.flx.gz")
    aSTD_grid_files  = glob.glob("marcs/ppl_vtur2_log*/*.flx.gz")
    
    a04_grid_files.sort()
    aNE_grid_files.sort()
    aSTD_grid_files.sort()
    
    grids = [a04_grid, aNE_grid, aSTD_grid]
    files = [a04_grid_files, aNE_grid_files, aSTD_grid_files]
    grid_names = ["a04_grid", "aNE_grid", "aSTD_grid"]
    
    for grid_i, (grid_files, grid) in enumerate(zip(files, grids)):
        # Load fluxes into grid, taking up to n_wl wavelength values
        print("-"*50, "\n", grid_names[grid_i], "\n", "-"*50)
        
        
        for gfile in grid_files:
            # Determine the parameters of the file
            gfn = gfile.split("/")[-1]
        
            try:
                if ":" in gfn:
                    temp_i = np.where(temps==int(gfn.split(":")[0][1:]))[0][0]
                    logg_i = np.where(loggs==float(gfn.split(":")[1][1:]))[0][0]
                    feh_i = np.where(fehs==float(gfn.split(":")[5][1:]))[0][0]
                else:
                    temp_i = np.where(temps==int(gfn.split("_")[0][1:]))[0][0]
                    logg_i = np.where(loggs==float(gfn.split("_")[1][1:]))[0][0]
                    feh_i = np.where(fehs==float(gfn.split("_")[5][1:]))[0][0]
                    
                print("[%i, %i, %i] - temp=%i, logg=%0.1f, Z=%0.2f" 
                      % (temp_i, logg_i, feh_i, temps[temp_i], loggs[logg_i], 
                         fehs[feh_i]))
            except:
                continue
            fluxes = np.loadtxt(gfile)
        
            grid[temp_i, logg_i, feh_i, :] = fluxes[:n_wl]
        
    return a04_grid, aNE_grid, aSTD_grid, temps, fehs, loggs


# -----------------------------------------------------------------------------
# Saving Photometry
# -----------------------------------------------------------------------------
def compute_photometry(a04_grid, aNE_grid, aSTD_grid, wl, temps, fehs, loggs):
    """Conpute photometry for the alpha enhancement grid cases specified.
    """
    photometry = []
    
    grids = [a04_grid, aNE_grid, aSTD_grid]
    logg_i = 0
    logg = 4.5
    
    # Boolean flags to indicate whether grid is enhanced, depleted, or standard
    alpha_bools = [[True, False, False], 
                   [False, True, False],
                   [False, False, True]]
    
    # Header and formatting specifications for saving the grid
    header = ("logg,[Fe/h],Teff,alpha_enh,alpha_dep,alpha_std,"
              "B,V,Rc,Ic,J,W,H,Ks,u,g,r,i,z,Y")
    fmt = ["%0.1f", "%0.2f", "%i", "%i", "%i", "%i", "%0.4f", "%0.4f","%0.4f",
           "%0.4f","%0.4f","%0.4f","%0.4f","%0.4f","%0.4f","%0.4f","%0.4f",
           "%0.4f","%0.4f","%0.4f"]
    
    # Loop over each grid and calculate synthetic photometry
    for grid_i, grid in enumerate(grids):
        for temp_i, temp in enumerate(temps):
            for logg_i, logg in enumerate(loggs):
                for feh_i, feh in enumerate(fehs):
                    # Get the fluxes for this portion of the grid, and check 
                    # the grid has actually been sampled here before proceeding
                    fluxes = grid[temp_i, logg_i, feh_i, :]
                
                    if np.sum(fluxes) > 1:
                        mags_ab = calculate_band_flux_ab(wl, fluxes)
                        mags_vega = calculate_band_flux_vega(wl, fluxes)
                
                        photometry.append([logg, feh, temp] 
                                           + alpha_bools[grid_i]
                                           + mags_vega.values() 
                                           + mags_ab.values())
    
    # Save and return the photometry            
    np.savetxt("photometry.csv", photometry, fmt=fmt, delimiter=",", 
               header=header)    
    
    return photometry
    
# -----------------------------------------------------------------------------
# Plotting Colour-Colour Diagrams
# -----------------------------------------------------------------------------
def plot_colour_colour_alpha(a04_grid, aNE_grid, aSTD_grid, wl, temps, fehs, 
                             loggs, x_bands=("i","Ks"), y_bands=("J","W")):
    """Plot J-W as a function of J-H with tracks of constant [Fe/H] and varying
    temp point colour, with fixed logg per plot.
    
    Note that grid has axes [temps, loggs, fehs, fluxes]
    """
    valid_filters = ["u", "g", "r", "i", "z", "Y", "B", "V", "R", "I", "J", 
                     "W", "H", "Ks"]
    
    # First check that these are valid colours
    if not np.all(np.isin([x_bands, y_bands], valid_filters)):
        raise Exception("Invalid Filters")
        
    alpha_treatment = ["Alpha Enhanced (alpha=0.4)", 
                       "Alpha Non-Enhanced (alpha=-0.4)",
                       "Standard Alpha: alpha=0.0 (Z=0.0), alpha=0.1 "
                       "(Z=-0.25), alpha=0.2 (Z=-0.5), alpha=0.3 (Z=-0.75), "
                       "alpha=0.4 (Z<=-1.0)"]
    grids = [a04_grid, aNE_grid, aSTD_grid]
              
    alpha_bools = [[True, False, False], 
                   [False, True, False],
                   [False, False, True]]
    
    plot_name = "plots/%s-%s_vs_%s-%s_alpha.pdf" % (x_bands[0], x_bands[1],
                                                    y_bands[0], y_bands[1])
    
    # Only iterate over a subset
    #temps = temps[::2]
    
    # Plot J-H versus J-W tracks for logg, [Fe/H], and Teff (fixed per plot)
    with PdfPages(plot_name) as pdf:
        for logg_i, logg in enumerate(loggs):
            for grid_i, grid in enumerate(grids):
                plt.close("all")
                do_plotting = False
                for temp_i, temp in enumerate(temps):
            
                    x_colour = []
                    y_colour = []
                    
                    for feh_i, feh in enumerate(fehs):
                        # Compute the magnitudes
                        fluxes = grid[temp_i, logg_i, feh_i, :]
                        
                        if np.sum(fluxes) > 1:
                            ab_mags = calculate_band_flux_ab(wl, fluxes)
                            vega_mags = calculate_band_flux_vega(wl, fluxes)
                            
                            # Get the magnitudes from the appropriate dict
                            if x_bands[0] in ab_mags:
                                xmag_1 = ab_mags[x_bands[0]]
                            else:
                                xmag_1 = vega_mags[x_bands[0]]
                                
                            if x_bands[1] in ab_mags:
                                xmag_2 = ab_mags[x_bands[1]]
                            else:
                                xmag_2 = vega_mags[x_bands[1]]
                                
                            if y_bands[0] in ab_mags:
                                ymag_1 = ab_mags[y_bands[0]]
                            else:
                                ymag_1 = vega_mags[y_bands[0]]
                                
                            if y_bands[1] in ab_mags:
                                ymag_2 = ab_mags[y_bands[1]]
                            else:
                                ymag_2 = vega_mags[y_bands[1]]
                                
                            # Calculate the colours
                            x_colour.append(xmag_1 - xmag_2)
                            y_colour.append(ymag_1 - ymag_2)
                            
                        else:
                            x_colour.append(np.nan)
                            y_colour.append(np.nan)

                    # Mask out any nan points in the series (due to the grid 
                    # having gaps) so we can plot continuous lines
                    x_colour = np.array(x_colour)
                    y_colour = np.array(y_colour)
                    mask = np.isfinite(x_colour)
                
                    # Plot lines and points separately to facilitate different
                    # colours and marker scales
                    if len(x_colour[mask]) > 0:
                        plt.scatter(x_colour, y_colour, s=128, c=fehs, 
                                    cmap="magma", zorder=2, vmin=-2, vmax=0.0)
                        plt.plot(x_colour[mask], y_colour[mask], "--", 
                                 zorder=1, label="Teff=%s" % temp)            
                        do_plotting = True
                    
            
                # Only bother with axis details/save plot if the grid had data
                # here, otherwise move on
                if do_plotting:
                    plt.title("[logg=%s] - %s" % (logg, 
                                                  alpha_treatment[grid_i]))
                    plt.xlabel("%s-%s" % x_bands)
                    plt.ylabel("%s-%s" % y_bands)
                    cb = plt.colorbar()
                    cb.set_label("[Fe/H]")
                    plt.clim(-2.0, 0.0)
                    plt.legend()
                    plt.grid()
                    plt.gcf().set_size_inches(16, 9)

                    pdf.savefig()
                plt.close() 


def plot_jw_vs_jh_fixed_temp(grid, wl, temps, fehs, loggs):
    """Plot J-W as a function of J-H with tracks of constant logg, [Fe/H] as 
    point colour, and fixed Teff per plot.
    
    Note that grid has axes [temps, loggs, fehs, fluxes]
    """
    # Define profiles
    w_band = np.array([[0.12*10**4, 0], [1.339*10**4, 0], [1.34*10**4, 1],
                       [1.5*10**4, 1], [1.501*10**4, 0], [21*10**4, 0]]) 
    filters = ["j", "h", "k"]
    [j_band, h_band, k_band] = [load_and_extend_profiles(filt) 
                                for filt in filters]
    
    # Plot J-H versus J-W tracks for logg, [Fe/H], and Teff (fixed per plot)
    with PdfPages("j-h_vs_j-w.pdf") as pdf:
        for temp_i, temp in enumerate(temps):
            plt.close("all")
            do_plotting = False
            
            for logg_i, logg in enumerate(loggs[::2]):
                j_h = []
                j_w = []
                marker_size = []
                for feh_i, feh in enumerate(fehs):
                    # Compute the magnitudes
                    j_mag = calculate_band_flux(wl[:60434], 
                                                grid[temp_i, logg_i, feh_i, :], 
                                                j_band, "J")
                    h_mag = calculate_band_flux(wl[:60434], 
                                                grid[temp_i, logg_i, feh_i, :], 
                                                h_band, "H")
                    w_mag = calculate_band_flux(wl[:60434], 
                                                grid[temp_i, logg_i, feh_i, :], 
                                                w_band, "W")

                    j_h.append(j_mag - h_mag)
                    j_w.append(j_mag - w_mag)
                    marker_size.append(logg)
                
                # Mask out any nan points in the series (due to the grid having
                # gaps) so we can plot continuous lines
                j_h = np.array(j_h)
                j_w = np.array(j_w)
                j_h_mask = np.isfinite(j_h)
                j_w_mask = np.isfinite(j_w)
                
                
                # Scale the marker sizes by gravity (i.e. giant stars have 
                # larger points)
                marker_size = 250 * 1/np.array(marker_size)**2
                
                # Plot lines and points separately to facilitate different
                # colours and marker scales
                if len(j_h[j_h_mask]) > 0:
                    plt.plot(j_h[j_h_mask], j_w[j_w_mask], "--", 
                             label="logg=%s" % logg)
                    plt.scatter(j_h, j_w, s=marker_size,  c=fehs, cmap="magma")
                                
                    do_plotting = True
            
            # Only bother with axis details/save plot if the grid had data at
            # this temperature - otherwise move on
            if do_plotting:
                plt.title(r"T$_{\rm eff} = $%i K" % temp)
                plt.xlabel("J-H")
                plt.ylabel("J-W")
                cb = plt.colorbar()
                cb.set_label("[Fe/H]")
                plt.legend()
                plt.grid()
                plt.gcf().set_size_inches(16, 9)

                pdf.savefig()
            plt.close() 
            
 
 
            
            
def plot_jw_vs_jh_fixed_logg(grid, wl, temps, fehs, loggs):
    """Plot J-W as a function of J-H with tracks of constant [Fe/H] and varying
    temp point colour, with fixed logg per plot.
    
    Note that grid has axes [temps, loggs, fehs, fluxes]
    """
    # Save jwhk data. Format: [logg, feh, teff, j, w, h]
    photometry = []
    
    # Define profiles
    w_band = np.array([[0.12*10**4, 0], [1.339*10**4, 0], [1.34*10**4, 1],
                       [1.5*10**4, 1], [1.501*10**4, 0], [21*10**4, 0]]) 
    filters = ["j", "h", "k"]
    [j_band, h_band, k_band] = [load_and_extend_profiles(filt) 
                                for filt in filters]
    
    # Plot J-H versus J-W tracks for logg, [Fe/H], and Teff (fixed per plot)
    with PdfPages("j-h_vs_j-w_fixed_logg.pdf") as pdf:
        for logg_i, logg in enumerate(loggs):
            plt.close("all")
            do_plotting = False
            for temp_i, temp in enumerate(temps[::2]):
            
                j_h = []
                j_w = []
                marker_size = []
                for feh_i, feh in enumerate(fehs):#[::2][:-1]):
                    # Compute the magnitudes
                    j_mag = calculate_band_flux(wl[:60434], 
                                                grid[temp_i, logg_i, feh_i, :], 
                                                j_band, "J")
                    h_mag = calculate_band_flux(wl[:60434], 
                                                grid[temp_i, logg_i, feh_i, :], 
                                                h_band, "H")
                    w_mag = calculate_band_flux(wl[:60434], 
                                                grid[temp_i, logg_i, feh_i, :], 
                                                w_band, "W")
                    
                    photometry.append([logg, feh, temp, j_mag, w_mag, h_mag])
                    
                    j_h.append(j_mag - h_mag)
                    j_w.append(j_mag - w_mag)
                
                # Mask out any nan points in the series (due to the grid having
                # gaps) so we can plot continuous lines
                j_h = np.array(j_h)
                j_w = np.array(j_w)
                j_h_mask = np.isfinite(j_h)
                j_w_mask = np.isfinite(j_w)
                
                # Plot lines and points separately to facilitate different
                # colours and marker scales
                if len(j_h[j_h_mask]) > 0:
                    #plt.scatter(j_h, j_w, s=128, c=temps[::2], cmap="magma", 
                    plt.scatter(j_h, j_w, s=128, c=fehs, cmap="magma", 
                                zorder=2)
                    plt.plot(j_h[j_h_mask], j_w[j_w_mask], "--", zorder=1,
                             label="Teff=%s" % temp)            
                    do_plotting = True
            
            # Only bother with axis details/save plot if the grid had data at
            # this temperature - otherwise move on
            if do_plotting:
                plt.title("logg = %0.1f" % logg)
                plt.xlabel("J-H")
                plt.ylabel("J-W")
                cb = plt.colorbar()
                cb.set_label("[Fe/H]")
                #cb.set_label(r"T$_{\rm eff}$ (K)")
                plt.legend()
                plt.grid()
                plt.gcf().set_size_inches(16, 9)

                pdf.savefig()
            plt.close() 
    
    # Save and return the colour information        
    np.savetxt("photometry.csv", photometry, delimiter=",", 
               header="logg,[Fe/h],Teff,Jmag,Wmag,Hmag")        
            
    return photometry
    

def plot_jw_vs_jh(grid, wl, temps, fehs, loggs, use_full_grid=False):
    """
    
    Note that grid has axes [temps, loggs, fehs, fluxes]
    """
    # Define profiles
    w_band = np.array([[0.12*10**4, 0], [1.339*10**4, 0], [1.34*10**4, 1],
                       [1.5*10**4, 1], [1.501*10**4, 0], [21*10**4, 0]]) 
    filters = ["j", "h", "k"]
    [j_band, h_band, k_band] = [load_and_extend_profiles(filt)
                                for filt in filters]
    
    plt.close("all")
    
    # Plot J-H versus J-W as a function of metalicity
    for feh_i, feh in enumerate(fehs):
        j_h = []
        j_w = []
        marker_size = []
        
        fixed_temp_1 = 3500
        fixed_logg_1 = 4.5
        
        fixed_temp_2 = 3500
        fixed_logg_2 = 2.5
        
        fixed_temp_3 = 3500
        fixed_logg_3 = 1.5
        
        for temp_i, temp in enumerate(temps):
            for logg_i, logg in enumerate(loggs):
                # Compute the magnitudes
                j_mag = calculate_band_flux(wl[:60434], 
                                            grid[temp_i, logg_i, feh_i, :], 
                                            j_band, "J")
                h_mag = calculate_band_flux(wl[:60434], 
                                            grid[temp_i, logg_i, feh_i, :], 
                                            h_band, "H")
                w_mag = calculate_band_flux(wl[:60434], 
                                            grid[temp_i, logg_i, feh_i, :], 
                                            w_band, "W")
                
                if ((temp == fixed_temp_1 and logg == fixed_logg_1)
                    or (temp == fixed_temp_2 and logg == fixed_logg_2)
                    or (temp == fixed_temp_3 and logg == fixed_logg_3)):
                    j_h.append(j_mag - h_mag)
                    j_w.append(j_mag - w_mag)
                    marker_size.append(logg)
                
                #if feh>=0:# and logg==5:
                    #plt.text(j_h[-1], j_w[-1], "%s K, logg=%s" % (temp, logg), 
                    #         fontsize="xx-small")
                    #print("...")
                #else:
                    #print("FeH=%s, logg=%s, teff=%s" % (feh, logg, temp), 
                    #      j_h[-1], j_w[-1])
        
        print("[Fe/H] = %s" % feh)
        marker_size = 100 * 1/np.array(marker_size)**2
        plt.scatter(j_h, j_w, label="[Fe/H]=%s" % feh, s=marker_size)
        
    plt.xlabel("J-H")
    plt.ylabel("J-W")
    plt.legend()
    plt.grid()
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("j-h_vs_j-w_vs_feh.pdf")


def plot_spectra(grid, wl, temps, fehs, loggs):
    """
    """
    plt.close("all")
    
    # Plot M4.5V
    #label = r"M4.5V T$_{\rm eff}$ = 3100 K, logg = 4.5, [Fe/H] = 0.75"
    #flux = grid[6, 10, 11]
    #plt.plot(wl[:60434], flux/np.max(flux), label=label, linewidth=0.05)
    
    # Plot M6.5V
    label = r"M4.5V T$_{\rm eff}$ = 2700 K, logg = 4.5, [Fe/H] = 0.0"
    flux = grid[6, 10, 8]
    plt.plot(wl[:60434], flux/np.max(flux), label=label, linewidth=0.05)
    
    # Plot M8V
    label = r"M4.5V T$_{\rm eff}$ = 2500 K, logg = 4.5, [Fe/H] = -1.0"
    flux = grid[6, 10, 4]
    plt.plot(wl[:60434], flux/np.max(flux), label=label, linewidth=0.05)
    
    # Plot M4.5III
    label = r"M4.5III T$_{\rm eff}$ = 3100 K, logg = 1.5, [Fe/H] = 0.75"
    flux = grid[6, 4, 11]
    plt.plot(wl[:60434], flux/np.max(flux), label=label, linewidth=0.05)
    
    # Plot the filter profiles
    plot_jhk_profiles()
    
    leg = plt.legend(loc="best")
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    
    plt.xlabel("Wavelength (A)")
    plt.ylabel("Normalised Flux")
    
    plt.xlim([10000, 20000])
    
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("w_band_spectra.pdf")


def plot_from_photometry():
    """
    """
    plt.close("all")
    # logg,[Fe/h],Teff,Jmag,Wmag,Hmag
    photometry = np.loadtxt("photometry.csv", delimiter=",")
    
    # Plot just logg=4.5
    phot_4d5 = photometry[photometry[:,0]==4.5, :]
    fehs = list(set(phot_4d5[:,1]))
    fehs.sort()
    temps = list(set(phot_4d5[:,2]))
    temps.sort()
    
    for temp in temps[::2]:
        subset = phot_4d5[phot_4d5[:,2]==temp, :]
        j_h = subset[:,3] - subset[:,5]
        j_w = subset[:,3] - subset[:,4]

        j_h = np.array(j_h)
        j_w = np.array(j_w)
        j_h_mask = np.isfinite(j_h)
        j_w_mask = np.isfinite(j_w)
        
        #import pdb
        #pdb.set_trace()
        plt.scatter(j_h, j_w, c=subset[:,1], cmap="magma")
        plt.plot(j_h[j_h_mask], j_w[j_w_mask], "--", label="Teff=%i" % temp)
        
        
    plt.xlabel("J-H")
    plt.ylabel("J-W")
    cb = plt.colorbar()
    cb.set_label("[Fe/H]")
    plt.legend()
    plt.grid()

"""
# Plot example fluxes with JHK filters overlaid
wl = np.loadtxt("/Users/adamrains/MARCS/flx_wavelengths.vac")
flx_cool = np.loadtxt(("/Users/adamrains/MARCS/flx/p3200_g+5.0_m0.0_t01_st_z+1.00"
                   "_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.flx"))

plt.close("all")        
plt.plot(wl, flx_cool, linewidth=0.1)
ax = plt.gca()

wb1 = 1.340 * 10**4
wb2 = 1.5 * 10**4
wb_bw = wb2-wb1
water_band = patches.Rectangle( (wb1, 0), wb_bw, 1*10**6, alpha=0.25, color="indianred")
ax.add_patch(water_band)

plot_jhk_profiles()
    
plt.xlabel("Wavelength (Angstroms)")
plt.ylabel(r"Intensity per unit wavelength (erg/cm$^2$/s/A)")
plt.xlim([1300, 25000])




# Determine magnitudes
filters = ["j", "h", "k"]
[j_band, h_band, k_band] = [load_and_extend_profiles(filt) for filt in filters]



# Vega fluxes
vega_j = 31.47 * 10**-11 # erg cm-2 s-1 A-1


flx_hot = np.loadtxt("/Users/adamrains/MARCS/MARCS_arains_20181214022648/p8000_g+4.0_m0.0_t02_st_z-0.50_a+0.20_c+0.00_n+0.00_o+0.20_r+0.00_s+0.00.flx")

flx_med = np.loadtxt("/Users/adamrains/MARCS/flx/p4000_g+4.0_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.flx")

flux_cool, mag_cool = calculate_band_flux(wl, flx_cool, j_band)
flux_med, mag_med = calculate_band_flux(wl, flx_med, j_band)
flux_hot, mag_hot = calculate_band_flux(wl, flx_hot, j_band)

print("3200 K", flux_cool, mag_cool)
print("4000 K", flux_med, mag_med)
print("8000 K", flux_hot, mag_hot)


#mean_flux_density = simp(

plt.plot(wl, flx_hot, linewidth=0.1)
plt.plot(wl, flx_med, linewidth=0.1)
"""




