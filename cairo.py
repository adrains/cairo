"""Code for planning a space-based water-band mission
"""

from __future__ import division, print_function

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from scipy.integrate import simps
from matplotlib.backends.backend_pdf import PdfPages


def load_and_extend_profiles(filter="j", min_wl=1200, max_wl=210000):
    """Load in the specified filter profile and zero pad both ends in 
    preparation for feeding into an interpolation function.
    """
    # Load the filter profile, and convert to Angstroms
    filt_profile = np.loadtxt("2mass_%s_profile.txt" % filter)
    filt_profile[:,0] = filt_profile[:,0] * 10**4
    
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
    
    r_sun = 6.95700*10**8        # metres
    d_10pc = 10 * 3.0857*10**16  # metres
    
    flux = (r_sun / d_10pc)**2 * fluxes
    
    mean_flux_density = (simps(flux*filt_profile*wavelengths, wavelengths)
                         / simps(filt_profile*wavelengths, wavelengths))
    
    vega_mags = {"j":[31.47* 10**-11, 0.899], 
                 "h":[11.38* 10**-11, 1.379], 
                 "k":[3.961* 10**-11, 1.886]}  # erg cm-2 s-1 A-1
    vega_mags["w"] = [np.mean([vega_mags["j"][0], vega_mags["h"][0]]),
                      np.mean([vega_mags["j"][1], vega_mags["h"][1]])]
    
    mag = -2.5 * np.log10(mean_flux_density/vega_mags[filt][0]) + vega_mags[filt][1]
    
    return mag
    

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
        filt_profile[:,1] = filt_profile[:,1] * plt.gca().get_ybound()[1]
        
        plt.fill_between(filt_profile[:,0], filt_profile[:,1], label=filt,
                         alpha=0.25, color=colours[filt_i])


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
                                                j_band, "j")
                    h_mag = calculate_band_flux(wl[:60434], 
                                                grid[temp_i, logg_i, feh_i, :], 
                                                h_band, "h")
                    w_mag = calculate_band_flux(wl[:60434], 
                                                grid[temp_i, logg_i, feh_i, :], 
                                                w_band, "w")

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
                                            j_band, "j")
                h_mag = calculate_band_flux(wl[:60434], 
                                            grid[temp_i, logg_i, feh_i, :], 
                                            h_band, "h")
                w_mag = calculate_band_flux(wl[:60434], 
                                            grid[temp_i, logg_i, feh_i, :], 
                                            w_band, "w")
                
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




