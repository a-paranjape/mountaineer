import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
#from plotSettings import setFigure
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from hmf import MassFunction
from scipy.optimize import fsolve, bisect
import cobaya
from cobaya import run
from scipy.optimize import minimize
import scipy

import reion_uvlf_funcs



def log_likelihood(lsum, ldiff, l2, l3, asum, adiff,  log10_fesc10, alpha_esc, log10Mcrit):
    """
    Routine to calculate the likelihood function, which will be maximised

    log {L} = - chi^2 / 2 


    i) Redshift evolution of log10_fstar10_by_cstar are decided by the parameters l0 , l1, l2 ,l3
    ii) Redshift evolution of alpha_star are decided by the parameters a0 , a1, a2 ,a3
    
    """
    
    a2 = l2
    a3 = l3 

    ################## Generate reionisation histroy for this particular choice of parameters ##########################
    
    # print("Fetching : model_QHI_z")    
    
    model_z_arr, model_QHII_arr, model_tau_value = reion_uvlf_funcs.reionHist_model(lsum, ldiff, l2, l3, asum, adiff, a2, a3, log10Mcrit, log10_fesc10, alpha_esc)

    ################## Build the model and data arrays for this particular choice of parameters ##########################
    
    model_QHI_UVLF_arr = []
    model_allUVLF_arr, data_allUVLF_arr, sigma_allUVLF_arr = reion_uvlf_funcs.model_and_data_allUVLF(model_QHI_UVLF_arr,log10Mcrit,lsum, ldiff, l2, l3, asum, adiff, a2, a3, log10_fesc10, alpha_esc)
    chisq_allUVLF = reion_uvlf_funcs.get_chisq(model_allUVLF_arr, data_allUVLF_arr, sigma_allUVLF_arr)
    
   
    model_QHI_arr, data_QHI_arr, sigma_QHI_arr = reion_uvlf_funcs.model_and_data_QHI(model_z_arr, model_QHII_arr)
    chisq_QHI = reion_uvlf_funcs.get_chisq(model_QHI_arr, data_QHI_arr, sigma_QHI_arr)
    
    model_tau_arr, data_tau_arr, sigma_tau_arr = reion_uvlf_funcs.model_and_data_tau_elec(model_tau_value)
    chisq_tau_elec = reion_uvlf_funcs.get_chisq(model_tau_arr, data_tau_arr, sigma_tau_arr)
    
    chisq_total = chisq_QHI + chisq_tau_elec + chisq_allUVLF
    #print (chisq_total)
    
    model_arr, data_arr, sigma_arr = reion_uvlf_funcs.model_and_data(lsum, ldiff, l2, l3, asum, adiff,  log10_fesc10, alpha_esc, log10Mcrit)
    chisq_total = reion_uvlf_funcs.get_chisq(model_arr, data_arr, sigma_arr)
    #print (chisq_total)
    
    
    

    ################## Build the log-likelihood function for this particular choice of parameters ##########################

    log_like = -0.5*chisq_total

    if not np.isfinite(log_like): log_like = -1.e6 ### handles blowing up of log-likelihood in case of infinities. We take logL = -1e6 as equiv to logL = -inf


    ################## Save the derived parameters for this particular choice of parameters ##########################
    
    
    l0=(lsum+ldiff)/2.0
    l1=(lsum-ldiff)/2.0

    a0=(asum+adiff)/2.0
    a1=(asum-adiff)/2.0

    
    twolone=2.0*l1          ### total jump in log10_fstar10_by_cstar
    twoaone=2.0*a1          ### total jump in alpha_star
    
    ## derived parameters ##

    derived = {"tau_e": model_tau_value, "twolone":twolone, "twoaone":twoaone, "l0" : l0, "l1" : l1, "a0" : a0, "a1" : a1}

    return log_like, derived

param_name_arr =  ["lsum","ldiff","l2","l3","asum","adiff","log10_fesc10", "alpha_esc","log10Mcrit"]
param_min_arr =   [-2.0,-2.0,8.0,0.5,0.0,0.0,-3.0,-3.0,9.0]
param_max_arr =   [ 2.0, 1.0,18.0,6.0,7.0,1.0,1.0,1.0,11.0]
param_start_arr = [-0.19, -0.92, 13.0, 2.1, 4.97, 0.34, -0.97, -0.41,9.32]
param_prop_arr =  [0.01,0.01, 0.01, 0.01,0.01, 0.01, 0.01, 0.01,0.01]
param_latex_name_arr = [r'\ell_{\varepsilon,0} + \dfrac{\ell_{\varepsilon,\mathrm{jump}}}{2}', r'\ell_{\varepsilon,0} - \dfrac{\ell_{\varepsilon,\mathrm{jump}}}{2}',r'z_\ast',r'\Delta z_\ast',r'\alpha_0 + \dfrac{\alpha_\mathrm{jump}}{2}',r'\alpha_0 - \dfrac{\alpha_\mathrm{jump}}{2}',r'\log_{10}~(\varepsilon_{\mathrm{esc,10}})',r'\alpha_{esc}',r'\log_{10}~M_{\mathrm{crit}}']
derived_name_arr = ["tau_e","twolone","twoaone","l0","l1","a0","a1"]
derived_min_arr =  [0.0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]
derived_max_arr =  [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]
derived_latex_name_arr = [r"\tau_{el}",r"\ell_{\varepsilon,\mathrm{jump}}",r"\alpha_\mathrm{jump}",r"\ell_{\varepsilon,0}",r"\ell_{\varepsilon,\mathrm{jump}} / 2",r"\alpha_0",r"\alpha_\mathrm{jump} / 2"]



##############  BEST_FIT MODEL ######################
fsum,fdiff,f2,f3,asum,adiff,log10_fesc10, alpha_esc,log10Mcrit  = -0.44444945, -0.87537585, 10.652879, 1.1410624, 0.78730058, 0.2882062, -0.81227004, -0.11191512, 10.228836
print(log_likelihood(fsum,fdiff,f2,f3,asum,adiff, log10_fesc10, alpha_esc, log10Mcrit)) 

#Returns : tau =  0.05475106
# Returns : chisq_allUVLF,chisq_QHI,chisq_tau_elec,chisq_total  =  48.42856294863958 5.222618536080999 0.011512128771554568 53.66269361349214
# Returns :  log_likelihood = -26.831346806745664




############### MCMC RUN CALL STATEMENT ######################


outdir = 'cobaya_chains/chains'
outfile = 'reion_uvlf_9p'
outloc = outdir+'/'+outfile
likelihood_name="fb_joint_LF_QHI"


info = reion_uvlf_funcs.create_cobaya_info_dict(log_likelihood, likelihood_name , outloc, param_name_arr, param_min_arr, param_max_arr, param_start_arr, param_prop_arr, derived_name_arr=derived_name_arr, derived_min_arr=derived_min_arr, derived_max_arr=derived_max_arr, param_latex_name_arr=param_latex_name_arr, derived_latex_name_arr=derived_latex_name_arr, only_minimize=False)
updated_info, sampler = cobaya.run(info,resume=True)

