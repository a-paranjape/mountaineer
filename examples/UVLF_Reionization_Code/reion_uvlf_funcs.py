import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
# from plotSettings import setFigure
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from hmf import MassFunction
from scipy.optimize import fsolve, bisect
import cobaya
from cobaya import run
from scipy.optimize import minimize
import scipy

# CONSTANTS AND UNIT CONVERSIONS


YR = 365*24*60*60      # seconds in a year
MYR = 1e6*YR              # seconds in a Myr
PC = 3.086e16             # metres in a parsec
MPC = 1e6*PC              # metres in a Mpc
tconv = (MPC/1e3)/MYR     # convert inverse Hubble to Myr

# fig = plt.figure(figsize=(10, 8))
# setFigure(fontsize=12)

### Cosmological Parameters ######

omega_m = 0.308
omega_l = 0.692
h = 0.678
omega_b = 0.0482
sigma_8 = 0.829
ns = 0.961
YHe = 0.24

#### More Useful Constants and Conversion Factors ######

mproton_cgs = 1.672623*1e-24           # in grams

Msun_cgs = 1.9891*1e33                  # in grams

mproton = mproton_cgs/Msun_cgs          # in solar mass

# print("Mass of proton in Msun = ", mproton)

Mpc_cm = 3.0856*1e24                    # conversion factor for Mpc ---> cm

# ρ_crit,0 = 2.7755 * 10^11 h^2 M_sun Mpc^{−3}

rho_crit = 2.7755 * (10**11)  				# units : h^2 Msun Mpc^{−3}

rho_m = omega_m*rho_crit*h*h			    # units : Msun Mpc^{−3}

rho_m_hunits = omega_m*rho_crit				# units : h^2 Msun Mpc^{−3}


n_H = (1-YHe)*(omega_b/omega_m)*(rho_m/mproton)  # units : Mpc^{−3}

n_H_cgs = n_H * (Mpc_cm)**(-3)  # units : cm^-3

# print("n_H_cgs = ",n_H_cgs)

sigmaT_cgs = 6.65246*1e-25  # units of cm^2

c_cgs = 2.99792458*1e10  # cm/s


#### Reionisation Model Parameters ######


# units of cm^3 s^{-1} - case B recombination coefficient of hydrogen at T_e = 1e4 K

alphaREC = 2.6*1e-13

eta_gamma_fid = 4.62175*1e60  # Msun^{-1}   - photons per Msolar

clumping = 3.0

#### Halo Mass Function Parameters ######

userhmf_model = 'SMT'

usertf_model = 'EH_BAO'

# Jenkins fit paramters to SMT mass function
userhmf_params = {"a": 0.73, "p": 0.175, "A": 0.353}


##### Luminosity Function Parameters ######

# SFR - UV Lum Conversion Factor

KUV_fid = 1.15485*(10**(-28))  # units : (M_sun/yr).(ergs/s/Hz)^-1


# list of reshifts at which UVLF obs. data is available for comparison to our model
zUVLF_arr = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.5, 13.2])

usercosmo = FlatLambdaCDM(H0=100*h, Om0=omega_m,
                          Ob0=omega_b, Tcmb0=2.725, Neff=0, m_nu=0)

##### Precompiled Tables ######

npz_file = np.load('HMF_Redshift_Table_JenkinsSMT.npz')

# 1D array of shape (201,): list of redshifts at which (dndm,m) has been evaluated
z_Table = npz_file['z_table']

# 1D array of shape (820,) : list of masses in units of Msun/h at which dndm has been evaluated
Mass_Table = npz_file['mass_table']

# 2D matrix of shape (201, 820): rows are individual redshifts ; columns are the dn/dm(m) values in units of h^4 Msun^{-1} Mpc^{-3} for values in Mass_Table
HMF_Table = npz_file['hmf_table']

############################# CODE STARTS FROM HERE ############################################


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ UV LUMINOSITY FUNCTION CALCULATION    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''


def get_log10_fstar_by_cstar_z(z, l0, l1, l2, l3):
    '''
    Equation 2.14 of arxiv:2404.02879
    '''

    return l0+l1*np.tanh((z-l2)/l3)


def get_alpha_star_z(z, a0, a1, a2, a3):
    '''
    Equation 2.15 of arxiv:2404.02879
    '''

    return a0+a1*np.tanh((z-a2)/a3)


def read_UVLF_data(filename):
    """
    Routine to read in UVLF observations data file
    """
    obsMUV_cen, obsPhiUV, obsPhiUV_hierr, obsPhiUV_loerr = np.loadtxt(
        filename, unpack=True, skiprows=2, usecols=(1, 2, 3, 4))
    obsDataset = np.loadtxt(filename, unpack=True,
                            skiprows=2, usecols=(0), dtype='U15')

    return obsMUV_cen, obsPhiUV, obsPhiUV_hierr, obsPhiUV_loerr, obsDataset


def read_QHI_data(filename):
    """
    Routine to read in QHI observations data file
    """
    obs_zarr, obs_xHI_arr, hi_sigmaxHI_arr, lo_sigmaxHI_arr = np.loadtxt(
        filename, unpack=True, skiprows=2, usecols=(1, 2, 3, 4))

    return obs_zarr, obs_xHI_arr, hi_sigmaxHI_arr, lo_sigmaxHI_arr


def Hub(z):
    """
    Routine to calculate the Hubble parameter at redshift z  ; units : (km/s) / Mpc

    """

    omega_k = 1 - omega_m - omega_l
    H0 = 100*h
    H_z = H0 * np.sqrt((omega_m)*(1 + z)**3 + omega_l + omega_k*(1 + z)**2)

    return H_z


def cal_MUV_neut_for_Mhalo(Mhalo, *params):
    """
    Routine to calculate galaxy luminosity for NEUTRAL REGIONS for a given halo mass and set of parameters

    Mhalo is in Msun and the supplied LUV is in ergs/s/Hz
    """

    # unpack the individual parameter values

    fstar10_by_cstar, alpha_star, z, Mcrit = params
    # print(params)

    # should have been just fstar10 * ((Mhalo/10**10)**alpha_star) here
    eps_star_by_cstar = fstar10_by_cstar * ((Mhalo/(10**10))**alpha_star)

    baryonfrac = omega_b/omega_m

    Mstar = eps_star_by_cstar * baryonfrac * Mhalo  # units of Msun

    # Calculate SFR timescale and SFR value

    tHub_z = 1.0/Hub(z)

    tHub_z = tHub_z * tconv   # Convert from units of inverse Hubble to Mega-year

    tHub_z = tHub_z * 1e6   # Convert from units of inverse Hubble to year

    # This should have been  tstar= cstar* tHub_z  but cstar already accounted for in Mstar  via "eps_star_by_cstar"
    tstar = tHub_z

    SFR = Mstar/tstar  # units : Msun/yr ;  the combination eps_star_by_cstar at the start in Mstar  NOW takes care of everything

    LUV_1500 = SFR/KUV_fid     # LUV in units of ergs/s/Hz

    # print("LUV_1500 = ",LUV_1500)

    MUV_1500 = -2.5*np.log10(LUV_1500)+51.6

    return MUV_1500


def cal_MUV_ion_for_Mhalo(Mhalo, *params):
    """
    Routine to calculate galaxy UV magnitude for IONIZED REGIONS for a given halo mass and set of parameters

    Mhalo is in Msun and the supplied LUV is in ergs/s/Hz

    """

    # unpack the individual parameter values

    fstar10_by_cstar, alpha_star, z, Mcrit = params
    # print("LUV_val = ",LUV_val)
    # Calculate stellar mass - Mstar

    # should have been just fstar10 * ((Mhalo/10**10)**alpha_star) here
    eps_star_by_cstar = fstar10_by_cstar * ((Mhalo/(10**10))**alpha_star)

    baryonfrac = omega_b/omega_m

    Mstar = eps_star_by_cstar * baryonfrac * Mhalo  # units of Msun

    # Calculate SFR timescale and SFR value

    tHub_z = 1.0/Hub(z)

    tHub_z = tHub_z * tconv   # Convert from units of inverse Hubble to Mega-year

    tHub_z = tHub_z * 1e6   # Convert from units of inverse Hubble to year

    # This should have been  tstar= cstar* tHub_z  but cstar already accounted for in Mstar  via "eps_star_by_cstar"
    tstar = tHub_z

    fgas = 2**(-Mcrit/Mhalo)

    # units : Msun/yr ;  the combination fstar10_by_cstar multiplied in the start in Mstar  NOW takes care of everything
    SFR = (fgas*Mstar)/tstar

    LUV_1500 = SFR/KUV_fid     # LUV in units of ergs/s/Hz

    # print("LUV_1500_ion : ",LUV_1500)

    # to handle errors while taking log of LUV_1500_ion
    LUV_1500_ion = np.where(LUV_1500 > 1e-3, LUV_1500, 1e-3)

    # MUV=150 is proxy for np.inf (i.e LUV_1500 < 1e-3)
    MUV_1500 = np.where(LUV_1500_ion > 1e-3, -2.5 *
                        np.log10(LUV_1500_ion)+51.6, 150)

    # print("MUV_1500_ion : ",MUV_1500)

    return MUV_1500


def find_dMh_by_dMUV_neut(Mhalo, MUV, fstar10_by_cstar, alpha_star, Mcrit, z):
    """
    Routine to find the derivative of halo mass w.r.t galaxy UV magnitude for NEUTRAL REGIONS

    The supplied Mhalo is in Msun and MUV is in mag

    """
    # Calculate SFR timescale and SFR value

    tHub_z = 1.0/Hub(z)  # should have been cstar * 1.0/Hub(z)  here

    tHub_z = tHub_z * tconv   # Convert from units of inverse Hubble to Mega-year

    tHub_z = tHub_z * 1e6   # Convert from units of inverse Hubble to year

    # Cosmoligical baryon fraction

    baryonfrac = omega_b/omega_m

    # neutral regions : Luv_neut =  (1/KUV_fid) *  (1/tstar)  * baryonfrac * [ fstar10 * ((Mhalo/(10**10))**alpha_star)) ] *  Mhalo

    # mass derivate of luminosity (dL/dM) ;  units :  (erg s^-1 Hz^-1).(M_sun)^-1
    dLUV_by_dMh = (fstar10_by_cstar/KUV_fid)*baryonfrac * (1/tHub_z)*(1+alpha_star) * \
        ((Mhalo/(10**10))**alpha_star)  # This is just (alpha_star+1)*LUV/Mh

    # units :  (erg s^-1 Hz^-1)^-1 (M_sun)
    dMh_by_dLUV = np.abs(1/dLUV_by_dMh)

    # derivative of AB magnitude wrt luminosity ; units : mag. (erg s^-1 Hz^-1)^-1
    # Remember : luv_neut =  ((fstar10_by_cstar/KUV_fid) * ((Mhalo/(10**10))**alpha_star)) * baryonfrac * Mhalo * (1/tHub_z)

    luv_neut = 10**(0.4*(51.6-MUV))

    dMUV_by_dLUV = np.abs(-2.5/(luv_neut*np.log(10)))

    dMh_by_dMUV = dMh_by_dLUV / dMUV_by_dLUV  # units :  M_sun . (mag)^-1

    return dMh_by_dMUV


def find_dMh_by_dMUV_ion(Mhalo, MUV, fstar10_by_cstar, alpha_star, Mcrit, z):
    """
    Routine to find the derivative of halo mass w.r.t galaxy UV magnitude for IONISED REGIONS

    The supplied Mhalo is in Msun and MUV is in mag

    """
    # Calculate SFR timescale and SFR value

    tHub_z = 1.0/Hub(z)  # should have been cstar * 1.0/Hub(z)  here

    tHub_z = tHub_z * tconv   # Convert from units of inverse Hubble to Mega-year

    tHub_z = tHub_z * 1e6   # Convert from units of inverse Hubble to year

    # Cosmoligical baryon fraction

    baryonfrac = omega_b/omega_m

    # mass derivate of luminosity (dL/dM) ;  units :  (erg s^-1 Hz^-1).(M_sun)^-1

    # Ionized regions : Luv_ion =  (1/KUV_fid) *  (1/tstar)  * baryonfrac * [ fstar10 * ((Mhalo/(10**10))**alpha_star)) ] *  Mhalo  * (2**(Mcrit/Mhalo))

    # Ionized regions : Luv_ion =  (1/KUV_fid) *  (1/tstar) * baryonfrac * f_star(Mhalo) * Mhalo  * fgas(Mhalo)

    # derivative of " f_star(Mhalo)*Mhalo " part

    prefix1 = (fstar10_by_cstar/KUV_fid) * baryonfrac * \
        (1/tHub_z) * (2**(-Mcrit/Mhalo))

    term1 = prefix1 * (1+alpha_star)*((Mhalo/(10**10))**alpha_star)

    # derivative of " fgas(Mhalo) " part
    prefix2 = (fstar10_by_cstar/KUV_fid) * baryonfrac * \
        (1/tHub_z) * ((Mhalo/(10**10)) ** alpha_star) * Mhalo

    term2 = prefix2 * np.log(2) * (Mcrit / (Mhalo**2)) * (2**(-Mcrit/Mhalo))

    dLUV_by_dMh = term1 + term2

    # units :  (erg s^-1 Hz^-1)^-1 (M_sun)
    dMh_by_dLUV = np.abs(1/dLUV_by_dMh)

    # derivative of AB magnitude wrt luminosity ; units : mag. (erg s^-1 Hz^-1)^-1

    # luv_ion =  ((fstar10_by_cstar/KUV_fid) * ((Mhalo/(10**10))**alpha_star)) * baryonfrac * Mhalo * (1/tHub_z) *(2**(Mcrit/Mhalo))

    luv_ion = 10**(0.4*(51.6-MUV))

    dMUV_by_dLUV = np.abs(-2.5/(luv_ion*np.log(10)))

    dMh_by_dMUV = dMh_by_dLUV / dMUV_by_dLUV  # units :  M_sun . (mag)^-1

    return dMh_by_dMUV


def UVLF_model(log10_fstar10_by_cstar, alpha_star, log10Mcrit, QHI, MUV_arr, z):
    """
    Routine to calculate the model UVLF in units of Mpc^-3 mag^-1 for each sample point in the parameter space

    """

    Mcrit = 10**log10Mcrit                          # units : Msun
    fstar10_by_cstar = 10**log10_fstar10_by_cstar  # units : unitless

    nbins = len(MUV_arr)

    # MUV_arr is array of MUV values at which PhiUV has been measured by observational surveys

    Phi_UV_total = np.empty(nbins)

    Phi_UV_neut = np.empty(nbins)
    Mhalo_UV_neut = np.empty(nbins)

    Phi_UV_ion = np.empty(nbins)
    Mhalo_UV_ion = np.empty(nbins)

    Mcool_z = get_Mcool(z)									    # units : Msun

    log10Mcool = np.log10(Mcool_z, dtype=np.float32)

    log10Mh_axis = np.arange(log10Mcool, 15, step=0.01)         # units : Msun
    # hmass_solution : Mcool(z) < Mhalo < 1e15 Msun

    fparams = (fstar10_by_cstar, alpha_star, z, Mcrit)

    # For the current set of parameters, create a spline interpolation function for the Mh-MUV relation in IONIZED and NEUTRAL regions

    MUV_neut_axis = cal_MUV_neut_for_Mhalo(10**log10Mh_axis, *fparams)
    # print(MUV_neut_axis)
    # print (log10Mh_axis)
    if (np.all(np.diff(MUV_neut_axis) < 0)):  # flip
        MhMUVneut_spline = scipy.interpolate.InterpolatedUnivariateSpline(
            x=np.flip(MUV_neut_axis), y=np.flip(log10Mh_axis), k=3, ext="zeros")
    elif (np.all(np.diff(MUV_neut_axis) > 0)):  # no flip
        MhMUVneut_spline = scipy.interpolate.InterpolatedUnivariateSpline(
            x=MUV_neut_axis, y=log10Mh_axis, k=3, ext="zeros")
    else:  # cannot deal with non monotonically increasing/decreasing functions
        return MUV_arr, 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins)
        # MUV_arr, Phi_UV_total, Phi_UV_neut, Phi_UV_ion, Mhalo_UV_neut, Mhalo_UV_ion

    MUV_ion_axis = cal_MUV_ion_for_Mhalo(10**log10Mh_axis, *fparams)
    # print(MUV_ion_axis)
    # do_flip_ion = True if np.all(np.diff(MUV_ion_axis) < 0) else False
    # print (np.all(np.diff(MUV_ion_axis) < 0), np.all(np.diff(MUV_ion_axis) > 0))
    # Before constructing interpolation step, crop out the (Mh,MUV) points where LUVion < 1e-3 (or, equivalently MUV==150) ; that arises due to the fgas factor for Mhalo above Mcool
    # This is needed to ensure monotonically strictly increasing x-axis for interpolation step

    if (len(np.argwhere(MUV_ion_axis == 150)) > 0):
        idx = np.argwhere(MUV_ion_axis == 150)[-1]
        ion_idx = idx[0]
        # print("ion_idx=",ion_idx)
        # print("len(MUV_ion_axis)=",len(MUV_ion_axis))
        # ion_picklist=np.linspace(ion_idx,len(MUV_ion_axis),num=len(MUV_ion_axis)-ion_idx+1)
        MUV_ion_axis_cropped = MUV_ion_axis[ion_idx:]
        log10Mh_axis_cropped = log10Mh_axis[ion_idx:]
        # print(MUV_ion_axis_cropped[1:10],MUV_ion_axis[1:10])
        # print(log10Mh_axis_cropped[1:10], log10Mh_axis[1:10])
        # print (np.all(np.diff(MUV_ion_axis_cropped) < 0), np.all(np.diff(MUV_ion_axis_cropped) > 0))

        if (np.all(np.diff(MUV_ion_axis_cropped) < 0)):  # flip
            MhMUVion_spline = scipy.interpolate.InterpolatedUnivariateSpline(x=np.flip(MUV_ion_axis_cropped), y=np.flip(log10Mh_axis_cropped), k=3, ext="zeros")
        elif (np.all(np.diff(MUV_ion_axis_cropped) > 0)):  # no flip
            MhMUVion_spline = scipy.interpolate.InterpolatedUnivariateSpline(x=MUV_ion_axis_cropped, y=log10Mh_axis, k=3, ext="zeros")
        else:  # cannot deal with non monotonically increasing/decreasing functions
            return MUV_arr, 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins)
    else:
        # print("ion_idx =  None")
        
        if (np.all(np.diff(MUV_ion_axis) < 0)):  # flip
            MhMUVion_spline = scipy.interpolate.InterpolatedUnivariateSpline(x=np.flip(MUV_ion_axis), y=np.flip(log10Mh_axis), k=3, ext="zeros")
        elif (np.all(np.diff(MUV_ion_axis) > 0)):  # no flip
            MhMUVion_spline = scipy.interpolate.InterpolatedUnivariateSpline(x=MUV_ion_axis, y=log10Mh_axis, k=3, ext="zeros")
        else:  # cannot deal with non monotonically increasing/decreasing functions
            return MUV_arr, 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins)
        

    ################## Build the luminosity function for this particular choice of paramters ##########################

    # flag variable to ensure that the HMF object is initialised only once (either in neutral/ionised section) , for rest of the times, just keep updating the mass attributes
    initialiseflag = 0

    for ibin in np.arange(nbins):

        M_UV_bin = MUV_arr[ibin]

        # print("Working for MUV = ",M_UV_bin)

        """
            ################################################################################################################################

            #################################       CALCULATE UV LUMINOSITY FUNCTION FOR NEUTRAL REGIONS  #################################

            ################################################################################################################################

            """

        Mh_UV_neut = 10**MhMUVneut_spline(M_UV_bin)  # units : Msun

        if (Mh_UV_neut > 1):  # i.e MhMUVneut_spline(M_UV_bin)>0

            # passing " Mh_UV*h " (which will be in units of Msun/h as Mh_UV is in units of Msun)
            log10Mmin_neut = np.log10(Mh_UV_neut*h)

            # done to ensure that HMF is queried with only one element in the mass array - which is = 10**Mh_UV_neut
            log10Mmax_neut = log10Mmin_neut+1e-4
            dlog10m = 0.1

            if initialiseflag == 0:
                # first-time initialise the HMF object
                HMF = MassFunction(z=z, Mmin=log10Mmin_neut, Mmax=log10Mmax_neut, dlog10m=dlog10m, sigma_8=sigma_8,
                                   n=ns, hmf_model=userhmf_model, hmf_params=userhmf_params, cosmo_model=usercosmo, transfer_model=usertf_model)
                initialiseflag = 1
            else:
                HMF.update(Mmin=log10Mmin_neut, Mmax=log10Mmax_neut)
                # the HMF object is already initialised above, just update the mass array attributes
            """
                m:                  : units Msun/h
                dndm                : units: h^4 Msun^{-1} Mpc^{-3}
                """

            # converting hmf.m values from Msun/h to Msun
            Mhalo_UV_neut[ibin] = (HMF.m[0])/h

            # converting hmf.dndm values from h^4 Msun^{-1} Mpc^{-3} to Msun^{-1} Mpc^{-3}
            dndm_MhaloUV_neut = (HMF.dndm[0]) * (h**4)

            # print("HMFCalc : Mhalo_UV[ibin] (in Msun) =",Mhalo_UV[ibin])
            # print("HMFCalc : dndm_Mhalo_UV = ",dndm_MhaloUV)

            dMhbydMUV_neut = find_dMh_by_dMUV_neut(
                Mhalo_UV_neut[ibin], M_UV_bin, fstar10_by_cstar, alpha_star, Mcrit, z)  # units :  Msun . (mag)^-1

            # units : Mpc^{-3} mag^{-1}  which comes from Msun^{-1} Mpc^{-3} * Msun (mag)^-1
            Phi_UV_neut[ibin] = dndm_MhaloUV_neut * dMhbydMUV_neut

        else:
            Mhalo_UV_neut[ibin] = 0.0
            Phi_UV_neut[ibin] = 0.0

        """
            ################################################################################################################################

            #################################       CALCULATE UV LUMINOSITY FUNCTION FOR IONISED REGIONS  #################################

            ################################################################################################################################

            """

        Mh_UV_ion = 10**MhMUVion_spline(M_UV_bin)  # units : Msun

        # print("Mh_UV (in Msun) =",Mh_UV)

        if (Mh_UV_ion > 1):  # i.e MhMUVion_spline(M_UV_bin) > 0

            # pass " Mh_UV*h " (which will be in units of Msun/h as Mh_UV is in units of Msun)
            log10Mmin_ion = np.log10(Mh_UV_ion*h)

            # done to ensure that HMF is queried with only one element in the mass array - which is = 10**Mh_UV_ion
            log10Mmax_ion = log10Mmin_ion+1e-4
            dlog10m = 0.1

            if initialiseflag == 0:
                # first-time initialise the HMF object (incase it didn't happen above for neutral regions)
                HMF = MassFunction(z=z, Mmin=log10Mmin_ion, Mmax=log10Mmax_ion, dlog10m=dlog10m, sigma_8=sigma_8, n=ns,
                                   hmf_model=userhmf_model, hmf_params=userhmf_params, cosmo_model=usercosmo, transfer_model=usertf_model)
                initialiseflag = 1
            else:
                HMF.update(Mmin=log10Mmin_ion, Mmax=log10Mmax_ion)
                # the HMF object is already initialised above, just update the mass array attributes

            # converting hmf.m values from Msun/h to Msun
            Mhalo_UV_ion[ibin] = (HMF.m[0])/h

            # converting hmf.dndm values from h^4 Msun^{-1} Mpc^{-3} to Msun^{-1} Mpc^{-3}
            dndm_MhaloUV_ion = (HMF.dndm[0]) * (h**4)

            # print("HMFCalc : Mhalo_UV[ibin] (in Msun) =",Mhalo_UV[ibin])
            # print("HMFCalc : dndm_Mhalo_UV = ",dndm_MhaloUV)

            dMhbydMUV_ion = find_dMh_by_dMUV_ion(
                Mhalo_UV_ion[ibin], M_UV_bin, fstar10_by_cstar, alpha_star, Mcrit, z)  # units :  Msun . (mag)^-1

            # units : Mpc^{-3} mag^{-1}  which comes from Msun^{-1} Mpc^{-3} * Msun (mag)^-1
            Phi_UV_ion[ibin] = dndm_MhaloUV_ion * dMhbydMUV_ion

        else:
            Mhalo_UV_ion[ibin] = 0.0
            Phi_UV_ion[ibin] = 0.0

        """
            ################################################################################################################################

            ######################   CALCULATE GLOBALLY AVG. TOTAL UV LUMINOSITY FUNCTION  #############################

            ################################################################################################################################

            """
        Phi_UV_total[ibin] = QHI*Phi_UV_neut[ibin] + \
            (1-QHI)*Phi_UV_ion[ibin]

    return MUV_arr, Phi_UV_total, Phi_UV_neut, Phi_UV_ion, Mhalo_UV_neut, Mhalo_UV_ion


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ REIONISATION HISTORY CALCULATION    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''


def dt_by_dz(z):
    """
    Routine to calculate dt/dz at redshift z  ; units : year
    """

    omega_k = 1 - omega_m - omega_l
    H0 = 100*h
    H_z = H0 * np.sqrt((omega_m)*(1 + z)**3 + omega_l + omega_k*(1 + z)**2)

    dtdz = -1.0/((1.0+z)*H_z)  # units : Mpc/ (km/s)

    dtdz = dtdz * tconv  # units : mega-year

    dtdz = dtdz * 1e6  # units : year

    return dtdz


############################################# [ REIONISATION ] ######################################

def chi(zargs):

    # assumes helium reionisation at z = 3

    if zargs >= 3.0:
        return 1.08
    else:
        return 1.16


def get_Mcool(zargs, Tvir=1.e4, mu=0.59):
    '''
    Returns minimum mass of haloes (in units of Msun) at redshift zargs where the gas can cool via atomic transitions and form stars
    '''
    omega_k = 1-omega_m-omega_l
    omega_m_z = omega_m*(1+zargs)**3/(omega_m*(1+zargs) **
                                      3+omega_l+omega_k*(1+zargs)**2)

    d = (omega_m_z-1)
    Delta_crit = 18.0*np.pi**2+82*d-39*d*d

    # Eq (25) of Barkana & Loeb (2001)

    Mcool = (1.e8 / h) * (Tvir / 1.98e4) ** 1.5 * (0.6 / mu) ** 1.5 * (omega_m_z /
                                                                       omega_m) ** (1./2.) * (18 * np.pi ** 2 / Delta_crit) ** (1./2.) * (10 / (1 + zargs)) ** 1.5

    return Mcool


def integral_neut(alpha_star, alpha_esc, zinp):

    Mcool_z = get_Mcool(zinp)									# Msun

    # converting units of Mcool_z to Msun/h
    log10Mcoolh = np.log10(Mcool_z*h)

    # find the index in the pre-computed redshift table which is closest to the passed redshift argument "zinp"
    index = np.argmin(np.abs(z_Table - zinp))

    # Units of Mass_table : Msun/h ; HMF_table : h^4 Msun^{-1} Mpc^{-3}

    mask = np.where(Mass_Table >= 10**log10Mcoolh)

    # dndm at redshift : zinp for masses > Mcool(zinp)
    dndm_z_arr = HMF_Table[index][mask]
    mass_arr = Mass_Table[mask]  # values of m >= Mcool(zinp)

    """
    mass_arr	                : units Msun/h
    dndm_z_arr	                : units: h^4 Msun^{-1} Mpc^{-3}

    """

    # units : Msun    ; array of halo masses greater than Mcool(zinp)
    m_arr = (mass_arr)/h

    # units : Msun^{-1} Mpc^{-3} ; array of dndm at zinp for halo masses greater than Mcool(zinp)
    dndm_arr = (dndm_z_arr) * (h**4)

    # quantity array for halo masses greater than Mcool(zinp)
    mpow_arr = np.power((m_arr/1e10), (alpha_star+alpha_esc))

    # integrand array contains halo masses greater than Mcool(zinp)

    integrand = mpow_arr*m_arr*dndm_arr

    integral = scipy.integrate.simpson(
        y=integrand, x=m_arr)  # units : Msun Mpc^{-3}

    return integral


def integral_ion(alpha_star, alpha_esc, zinp, Mcrit):

    Mcool_z = get_Mcool(zinp)									# Msun

    # converting units of Mcool_z to Msun/h
    log10Mcoolh = np.log10(Mcool_z*h)

    # find the index in the pre-computed redshift table which is closest to the passed redshift argument "zinp"
    index = np.argmin(np.abs(z_Table - zinp))

    # Units of Mass_table : Msun/h ; HMF_table : h^4 Msun^{-1} Mpc^{-3}

    mask = np.where(Mass_Table >= 10**log10Mcoolh)

    # values of dndm at the redshift z=zinp for masses >= Mcool(zinp)
    dndm_z_arr = HMF_Table[index][mask]
    mass_arr = Mass_Table[mask]  # values of m >= Mcool(zinp)

    """
    mass_arr	                : units Msun/h
    dndm_z_arr	                : units: h^4 Msun^{-1} Mpc^{-3}

    """

    # units : Msun    ; array of halo masses greater than Mcool(zinp)
    m_arr = (mass_arr)/h

    # units : Msun^{-1} Mpc^{-3} ; array of dndm at zinp for halo masses greater than Mcool(zinp)
    dndm_arr = (dndm_z_arr) * (h**4)

    # quantity array for halo masses greater than Mcool(zinp)
    mpow_arr = np.power((m_arr/1e10), (alpha_star+alpha_esc))

    # print("ion : ", mpow_arr.shape)
    # gas fraction array for halo masses greater than Mcool(zinp)
    fgas_arr = np.power(2.0, -1.0*np.divide(Mcrit, m_arr))
    # print("ion : ", fgas_arr.shape)

    # integrand array contains halo masses greater than Mcool(zinp)

    integrand = fgas_arr*mpow_arr*m_arr*dndm_arr

    integral = scipy.integrate.simpson(
        y=integrand, x=m_arr)  # units : Msun Mpc^{-3}

    return integral


def niondot_by_nH_neut(fstar10_by_cstar, fesc10, alpha_star, alpha_esc, zinp):

    tHub = 1.0/Hub(zinp)  # units : Mpc / (km/s)

    tHub = tHub * tconv  # units : mega-years

    tHub = tHub * 1e6  # units : year

    prefactor = (fstar10_by_cstar * fesc10 * eta_gamma_fid * mproton) / \
        ((1-YHe)*tHub)  # year ^-1

    # units : Msun Mpc^{-3}  --- same units as rho_m
    intg_val_neut = integral_neut(alpha_star, alpha_esc, zinp)

    # print("neut : ",prefactor*(intg_val_neut/rho_m))

    return prefactor*(intg_val_neut/rho_m)  # year^-1


def niondot_by_nH_ion(fstar10_by_cstar, fesc10, alpha_star, alpha_esc, zinp, Mcrit):

    tHub = 1.0/Hub(zinp)  # units : Mpc / (km/s)

    tHub = tHub * tconv  # units : mega-years

    tHub = tHub * 1e6  # units : year

    prefactor = (fstar10_by_cstar * fesc10 * eta_gamma_fid * mproton) / \
        ((1-YHe)*tHub)  # year^-1

    # units : Msun Mpc^{-3}  --- same units as rho_m
    intg_val_ion = integral_ion(alpha_star, alpha_esc, zinp, Mcrit)

    # print("ion : ",prefactor*(intg_val_ion/rho_m))

    return prefactor*(intg_val_ion/rho_m)  # units : year^-1


def get_QII_arr(z_arr, l0, l1, l2, l3, a0, a1, a2, a3, fesc10, alpha_esc, Mcrit):
    '''
    Implementation of numerical derivative is as per Euler Backward Method 
    '''

    QII_arr = np.empty(len(z_arr))
    # for num=201, dz = 0.1 and for num = 401, dz = 0.05
    dz = np.float32(z_arr[1]-z_arr[0])
    QII_arr[0] = 0  # QII at z_arr[0] (z = 20) is taken to be 0.0

    for i, z in enumerate(z_arr[:-1]):

        if (QII_arr[i] >= 1.0):

            QII_arr[i] = 1.0
            QII_arr[i+1] = 1.0

        else:
            log10fstar10_by_cstar = get_log10_fstar_by_cstar_z(
                z_arr[i+1], l0, l1, l2, l3)

            fstar10_by_cstar = 10.0**log10fstar10_by_cstar

            alpha_star = get_alpha_star_z(z_arr[i+1], a0, a1, a2, a3)

            numerator_term = QII_arr[i] + dz*dt_by_dz(z_arr[i+1])*niondot_by_nH_neut(
                fstar10_by_cstar, fesc10, alpha_star, alpha_esc, z_arr[i+1])  # unitless

            # n_H_cgs : cm^{−3} and (alphaREC*YR) : cm^3 yr^{-1} as alphaREC : cm^3 sec^{-1}

            recomb_term = chi(z_arr[i+1])*n_H_cgs * ((1.0+z_arr[i+1])
                                                     ** 3.0) * alphaREC * YR * clumping  # year^-1

            denominator_term = 1.0-(dz*dt_by_dz(z_arr[i+1])*(niondot_by_nH_ion(fstar10_by_cstar, fesc10, alpha_star, alpha_esc, z_arr[i+1], Mcrit) - niondot_by_nH_neut(
                fstar10_by_cstar, fesc10, alpha_star, alpha_esc, z_arr[i+1])-recomb_term))  # year * year^-1 = unitless

            QII_arr[i+1] = numerator_term/denominator_term

    return QII_arr


def tau_integrand(QII_arr, z_arr):

    chi_arr = [chi(z) for z in z_arr]
    # units : Mpc / (km/s)  ----> Myr ----> secs
    invHz_arr = [(1.0/Hub(z))*tconv*MYR for z in z_arr]

    intgrnd = chi_arr*QII_arr * \
        np.power((1.0+z_arr), 2)*invHz_arr  # units : secs

    return intgrnd


def get_tau(QII_arr, z_arr):

    # z_arr goes from 20 to 0 and QII_arr goes from 0 to 1
    tau_intgrnd = tau_integrand(QII_arr, z_arr)

    integral = scipy.integrate.simpson(y=tau_intgrnd, x=z_arr)  # units : sec

    # c_cgs : cm / sec  ; n_H_cgs : cm^-3 ; sigmaT_cgs : cm^2 ; integrand : sec

    # unitless - take absolute value to get rid of hassle of integration limits
    return np.abs(c_cgs*n_H_cgs*sigmaT_cgs*integral)


def reionHist_model(lsum, ldiff, l2, l3, asum, adiff, a2, a3, log10Mcrit, log10_fesc10, alpha_esc):

    l0 = (lsum+ldiff)/2.0
    l1 = (lsum-ldiff)/2.0

    a0 = (asum+adiff)/2.0
    a1 = (asum-adiff)/2.0

    fesc10 = 10.0**log10_fesc10
    Mcrit = 10.0**log10Mcrit

    z_arr = np.linspace(20.0, 0.0, num=201)

    model_QII_arr = get_QII_arr(
        z_arr, l0, l1, l2, l3, a0, a1, a2, a3, fesc10, alpha_esc, Mcrit)

    # z_end = z_arr[np.argmax(QII_arr == 1.0)]

    tau_model = get_tau(model_QII_arr, z_arr)

    return z_arr, model_QII_arr, tau_model

 #########################################################################################


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ LIKELIHOOD CALCULATIONS    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''


def create_cobaya_info_dict(likelihood_func, likelihood_name, outroot, param_name_arr, param_min_arr, param_max_arr, param_start_arr, param_prop_arr, derived_name_arr=None, derived_min_arr=None, derived_max_arr=None, param_latex_name_arr=None, derived_latex_name_arr=None, only_minimize=False, Rminus1_stop=0.01, Rminus1_cl_stop=0.1, max_tries=10000):

    ndim = len(param_name_arr)
    nderived = 0
    info = {}
    info["likelihood"] = {}
    info["likelihood"][likelihood_name] = {"external": likelihood_func}
    info["likelihood"][likelihood_name]["input_params"] = param_name_arr
    if (derived_name_arr is not None):
        nderived = len(derived_name_arr)
        info["likelihood"][likelihood_name]["output_params"] = derived_name_arr

    info["params"] = {}
    for i in range(ndim):
        info["params"][param_name_arr[i]] = {"prior": {
            "min": param_min_arr[i], "max": param_max_arr[i]}, "ref": param_start_arr[i], "proposal": param_prop_arr[i]}
        if (param_latex_name_arr is not None):
            info["params"][param_name_arr[i]]["latex"] = param_latex_name_arr[i]

    for i in range(nderived):
        info["params"][derived_name_arr[i]] = {}
        if (derived_min_arr is not None and derived_max_arr is not None):
            info["params"][derived_name_arr[i]] = {
                "min": derived_min_arr[i], "max": derived_max_arr[i]}
        if (derived_latex_name_arr is not None):
            info["params"][derived_name_arr[i]
                           ]["latex"] = derived_latex_name_arr[i]

    if (only_minimize):
        info["sampler"] = {"minimize": {"method": "bobyqa", "override_bobyqa": {
            "seek_global_minimum": True, "rhoend": 1.e-8}}}  # the scipy solver does not work
    else:
        info["sampler"] = {"mcmc": {"Rminus1_stop": Rminus1_stop,
                                    "max_tries": max_tries, "Rminus1_cl_stop": Rminus1_cl_stop}}
    info["output"] = outroot
    info["debug"] = 'debug.txt'

    return info


def model_and_data_tau_elec(model_tau):
    """

    Routine to compute the chi-square for the observable : Electron Scattering Optical Depth (tau_el)

    """

    tau_e_obs = 0.054
    sigma_tau_e_obs = 0.007

    return np.array([model_tau]), np.array([tau_e_obs]), np.array([sigma_tau_e_obs])


def model_and_data_QHI(z_arr, QHII_arr):
    """

    Routine to compute the chi-square for the observable : neutral hydrogen fraction (Q_HI)

    """

    QHI_arr = 1.0-QHII_arr

    model_arr = []
    data_arr = []
    sigma_arr = []

    ############################################################################################################################
    ###########      Working with data points which are MEASUREMENTS of Q_HI(z)                   ########################
    ############################################################################################################################

    obs_zarr, obs_xHI_arr, hi_sigmaxHI_arr, lo_sigmaxHI_arr = read_QHI_data(
        'QHI_datafiles/fullQHIdata.txt')

    for i, z_obs in enumerate(obs_zarr):

        if (lo_sigmaxHI_arr[i] < 0.0):
            mean_obs = obs_xHI_arr[i] + 0.5 * \
                (hi_sigmaxHI_arr[i] + lo_sigmaxHI_arr[i])
            sigma = 0.5 * (hi_sigmaxHI_arr[i] - lo_sigmaxHI_arr[i])
        else:
            mean_obs = obs_xHI_arr[i] + 0.5 * \
                (hi_sigmaxHI_arr[i] - lo_sigmaxHI_arr[i])
            sigma = 0.5 * (hi_sigmaxHI_arr[i] + lo_sigmaxHI_arr[i])

        # find the index in model redshift array closest to the observed redshift
        indx = np.argmin(np.abs(z_obs - z_arr))

        modelz = z_arr[indx]
        model = QHI_arr[indx]

        model_arr = np.append(model_arr, model)
        data_arr = np.append(data_arr, mean_obs)
        sigma_arr = np.append(sigma_arr, sigma)

    return model_arr, data_arr, sigma_arr


def model_and_data_eachUVLF(log10_fstar10_by_cstar, alpha_star, log10Mcrit, QHI_zval, zval, obsdatafile):
    """

    Routine to compute the chi-square for each UV Luminosity Function (UVLF) at a given redshift "zval"

    """

    ################## Obtain the observational UVLF data for this particular redshift ##########################

    obsMUV_cen, obsPhiUV, obsPhiUV_hierr, obsPhiUV_loerr, obsDataset = read_UVLF_data(
        obsdatafile)

    ################## Obtain the model UVLF for this particular choice of parameters ##########################

    modelMUV_cen, modelUVLF_total, modelUVLF_neut, modelUVLF_ion, modelMhaloUV_neut, modelMhaloUV_ion = UVLF_model(
        log10_fstar10_by_cstar, alpha_star, log10Mcrit, QHI_zval, obsMUV_cen, zval)

    ################## Build the chi squared function for this particular choice of parameters ##########################

    model_arr = []
    data_arr = []
    sigma_arr = []

    for i, obsMUV in enumerate(obsMUV_cen):

        if (obsPhiUV_loerr[i] < 0.0):
            mean_obs = obsPhiUV[i] + 0.5 * \
                (obsPhiUV_hierr[i] + obsPhiUV_loerr[i])
            sigma = 0.5 * (obsPhiUV_hierr[i] - obsPhiUV_loerr[i])
        else:
            mean_obs = obsPhiUV[i] + 0.5 * \
                (obsPhiUV_hierr[i] - obsPhiUV_loerr[i])
            sigma = 0.5 * (obsPhiUV_hierr[i] + obsPhiUV_loerr[i])

        model = modelUVLF_total[i]

        model_arr = np.append(model_arr, model)
        data_arr = np.append(data_arr, mean_obs)
        sigma_arr = np.append(sigma_arr, sigma)

    return model_arr, data_arr, sigma_arr


def model_and_data_allUVLF(QHI_UVLF_arr, log10Mcrit, lsum, ldiff, l2, l3, asum, adiff, a2, a3, log10_fesc10, alpha_esc):
    """

    Routine to compute the chi-square for the observable : ALL UV Luminosity Functions (UVLF) in the range  6.0 <= z <= 13.2

    """
    l0 = (lsum+ldiff)/2.0
    l1 = (lsum-ldiff)/2.0

    a0 = (asum+adiff)/2.0
    a1 = (asum-adiff)/2.0

    model_arr = []
    data_arr = []
    sigma_arr = []

    model_z_arr, model_QHII_arr, model_tau_value = reionHist_model(
        lsum, ldiff, l2, l3, asum, adiff, a2, a3, log10Mcrit, log10_fesc10, alpha_esc)

    model_QHI_UVLF_arr = np.empty(len(zUVLF_arr))

    for i, zdata in enumerate(zUVLF_arr):

        # log10Mcrit=get_log10Mcrit_z(zdata)

        loop_indx = np.argmin(np.abs(zdata - model_z_arr))

        QHI = 1 - model_QHII_arr[loop_indx]

        # contains he QHI values at the redshifts where the model predicted UVLF will be compared with the data
        model_QHI_UVLF_arr[i] = QHI

        log10fstar10_by_cstar = get_log10_fstar_by_cstar_z(
            zdata, l0, l1, l2, l3)

        alpha_star = get_alpha_star_z(zdata, a0, a1, a2, a3)

        # QHI_UVLF_arr contains the QHI values at the redshifts where UVLF will be compared with the data (z = z_UVLF_arr)
        # QHI = QHI_UVLF_arr[i]
        QHI = model_QHI_UVLF_arr[i]
        if zdata == 5.0:
            obsdatafile = 'UVLF_datafiles/UVLF_z5p0.txt'
        elif zdata == 6.0:
            obsdatafile = 'UVLF_datafiles/UVLF_z6p0.txt'
        elif zdata == 7.0:
            obsdatafile = 'UVLF_datafiles/UVLF_z7p0.txt'
        elif zdata == 8.0:
            obsdatafile = 'UVLF_datafiles/UVLF_z8p0.txt'
        elif zdata == 9.0:
            obsdatafile = 'UVLF_datafiles/UVLF_z9p0.txt'
        elif zdata == 10.0:
            obsdatafile = 'UVLF_datafiles/UVLF_z10p0.txt'
        elif zdata == 11.0:
            obsdatafile = 'UVLF_datafiles/UVLF_z11p0.txt'
        elif zdata == 12.5:
            obsdatafile = 'UVLF_datafiles/UVLF_z12p5.txt'
        elif zdata == 13.2:
            obsdatafile = 'UVLF_datafiles/UVLF_z13p2.txt'

        model_eachUVLF_arr, data_eachUVLF_arr, sigma_eachUVLF_arr = model_and_data_eachUVLF(
            log10fstar10_by_cstar, alpha_star, log10Mcrit, QHI, zdata, obsdatafile)
        model_arr = np.append(model_arr, model_eachUVLF_arr)
        data_arr = np.append(data_arr, data_eachUVLF_arr)
        sigma_arr = np.append(sigma_arr, sigma_eachUVLF_arr)

    # outside for-loop
    return model_arr, data_arr, sigma_arr


def get_chisq(model_arr, data_arr, sigma_arr):

    return np.sum(((data_arr - model_arr) / sigma_arr) ** 2)


def model_and_data(lsum, ldiff, l2, l3, asum, adiff,  log10_fesc10, alpha_esc, log10Mcrit):

    # TIE UP the transition redshift and transition redshift interval for log10_fstar10_by_cstar and alpha_star together

    model_arr = []
    data_arr = []
    sigma_arr = []

    a2 = l2
    a3 = l3

    ################## Generate reionisation histroy for this particular choice of parameters ##########################

    # print("Fetching : model_QHI_z")

    model_z_arr, model_QHII_arr, model_tau_value = reionHist_model(
        lsum, ldiff, l2, l3, asum, adiff, a2, a3, log10Mcrit, log10_fesc10, alpha_esc)

    model_QHI_UVLF_arr = np.empty(len(zUVLF_arr))

    # print("Obtained : model_QHI_z")

    for i, zdata in enumerate(zUVLF_arr):

        # find the index closest to the observed redshift
        loop_indx = np.argmin(np.abs(zdata - model_z_arr))

        QHI = 1 - model_QHII_arr[loop_indx]

        # contains he QHI values at the redshifts where the model predicted UVLF will be compared with the data
        model_QHI_UVLF_arr[i] = QHI

    ################## Build the model and data arrays for this particular choice of parameters ##########################

    model_allUVLF_arr, data_allUVLF_arr, sigma_allUVLF_arr = model_and_data_allUVLF(
        model_QHI_UVLF_arr, log10Mcrit, lsum, ldiff, l2, l3, asum, adiff, a2, a3, log10_fesc10, alpha_esc)
    model_arr = np.append(model_arr, model_allUVLF_arr)
    data_arr = np.append(data_arr, data_allUVLF_arr)
    sigma_arr = np.append(sigma_arr, sigma_allUVLF_arr)

    model_QHI_arr, data_QHI_arr, sigma_QHI_arr = model_and_data_QHI(
        model_z_arr, model_QHII_arr)
    model_arr = np.append(model_arr, model_QHI_arr)
    data_arr = np.append(data_arr, data_QHI_arr)
    sigma_arr = np.append(sigma_arr, sigma_QHI_arr)

    model_tau_arr, data_tau_arr, sigma_tau_arr = model_and_data_tau_elec(
        model_tau_value)
    model_arr = np.append(model_arr, model_tau_arr)
    data_arr = np.append(data_arr, data_tau_arr)
    sigma_arr = np.append(sigma_arr, sigma_tau_arr)

    return model_arr, data_arr, sigma_arr
