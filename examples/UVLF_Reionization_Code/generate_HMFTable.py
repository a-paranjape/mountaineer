import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
from plotSettings import setFigure
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from hmf import MassFunction


# CONSTANTS AND UNIT CONVERSIONS


YR = 365*24*60*60      # seconds in a year
MYR = 1e6*YR              # seconds in a Myr
PC = 3.086e16             # metres in a parsec
MPC = 1e6*PC              # metres in a Mpc
tconv = (MPC/1e3)/MYR     # convert inverse Hubble to Myr

fig = plt.figure(figsize=(10, 8))
setFigure(fontsize=12)

### Cosmological Parameters ######

omega_m = 0.308
omega_l = 0.692
h = 0.678
omega_b = 0.0482
sigma_8 = 0.829
ns = 0.961
YHe = 0.24

usercosmo = FlatLambdaCDM(H0=100*h, Om0=omega_m,Ob0=omega_b, Tcmb0=2.725, Neff=0, m_nu=0)

#### Halo Mass Function Parameters ######

## all mass units below are in Msun/h

log10Mmin=6.8 
log10Mmax=15.0
dlog10m=0.01   

# so 820 points along the mass axis ; since np.arange is used 


userhmf_model = 'SMT'
usertf_model = 'EH_BAO'
userhmf_params={"a":0.73, "p":0.175, "A":0.353}

z_arr = np.linspace(20.0, 0.0, num=201,dtype=np.float32)

print(z_arr)
print(z_arr.shape)

hmf_z_arr=np.zeros(shape=(len(z_arr),820))

mass_arr=np.zeros(820)


for iz, z_val in enumerate(z_arr): 
    print("z = ",z_val)
    
    HMF = MassFunction(z=z_val, Mmin=log10Mmin, Mmax=log10Mmax, dlog10m=dlog10m, sigma_8=sigma_8,n=ns, hmf_model=userhmf_model,hmf_params=userhmf_params, cosmo_model=usercosmo, transfer_model=usertf_model)
	
    if (iz==0):
        mass_arr=HMF.m 			#units Msun/h
		
    hmf_z_arr[iz]=HMF.dndm		#units: h^4 Msun^{-1} Mpc^{-3}
	

print(HMF.m.shape)

npzfilename="HMF_Redshift_Table_JenkinsSMT.npz"

np.savez(npzfilename,z_table=z_arr,mass_table=mass_arr,hmf_table=hmf_z_arr)



