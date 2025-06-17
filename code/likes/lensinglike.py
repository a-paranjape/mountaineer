import jax
import jax.numpy as np
import jax_cosmo as jc
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory

N_Params_Tot = 16

from astropy.io import fits

#########################################
class LensingTheory(Theory):
    n_params = 16
    fixed_params = {}
    ell = np.logspace(1, 3)
    nz_source = fits.getdata('data/2pt_NG_mcal_1110.fits', 6)
    # This is the effective number of sources from the cosmic shear paper
    neff_s = [1.47, 1.46, 1.50, 0.73]
    #########################################
    def initialize(self):
        self.nzs_s = [jc.redshift.kde_nz(self.nz_source['Z_MID'].astype('float32'),
                                         self.nz_source['BIN%d'%i].astype('float32'), 
                                         bw=0.01,
                                         gals_per_arcmin2=self.neff_s[i-1])
                      for i in range(1,5)]
            
        n_tot = self.n_params + len(self.fixed_params.keys())
        if n_tot != N_Params_Tot:
            raise Exception('LensingTheory works with 16 params, but {0:d} specified.'.format(n_tot))
            
        self.all_params = {'sig8':0.0,'Oc':0.0,'Ob':0.0,'h':0.0,'ns':0.0,'w0':0.0} # re-usable dictionary
        for m in range(1,5):
            self.all_params['m{0:d}'.format(m)] = 0.0
        for dz in range(1,5):
            self.all_params['dz{0:d}'.format(dz)] = 0.0
        self.all_params['A'] = 0.0
        self.all_params['eta'] = 0.0
        self.varied_params = list(self.all_params.keys())
        for par in self.fixed_params.keys():
            self.varied_params.remove(par)
            self.all_params[par] = self.fixed_params[par]
        # now self.varied_params is list of names of varied parameters and self.all_params is updated with fixed_params values
        print('varied params: ',self.varied_params)
        
        self.mu = jax.jit(self.mu_slow)
        
    #########################################


    #########################################
    def unpack_params_vec(self,params):
        # Retrieve cosmology
        cosmo = jc.Cosmology(sigma8=params[0], Omega_c=params[1], Omega_b=params[2],
                             h=params[3], n_s=params[4], w0=params[5],
                             Omega_k=0., wa=0.)
        m1,m2,m3,m4 = params[6:10]
        dz1,dz2,dz3,dz4 = params[10:14]
        A = params[14]
        eta = params[15]
        return cosmo, [m1,m2,m3,m4], [dz1,dz2,dz3,dz4], [A, eta]
    #########################################

    #########################################
    def mu_slow(self,params):
        # First unpack parameter vector
        cosmo, m, dz, (A, eta) = self.unpack_params_vec(params) 

        # Build source nz with redshift systematic bias
        nzs_s_sys = [jc.redshift.systematic_shift(nzi, dzi) 
                    for nzi, dzi in zip(self.nzs_s, dz)]

        # Define IA model, z0 is fixed
        b_ia = jc.bias.des_y1_ia_bias(A, eta, 0.62)

        # Define the lensing and number counts probe
        probes = [jc.probes.WeakLensing(nzs_s_sys, 
                                        ia_bias=b_ia,
                                        multiplicative_bias=m)]

        cl = jc.angular_cl.angular_cl(cosmo, self.ell, probes)

        return cl
    #########################################

    
    #########################################
    def calculate(self,state, want_derived=False, **param_dict):
        keys = list(param_dict.keys())
        if keys != self.varied_params:
            raise Exception('mismatched keys in LensingTheory.calculate')

        p_this = list(param_dict.values()) # self.params.T[0].copy()
        p = 0
        for key in keys: 
            self.all_params[key] = p_this[p]
            p += 1
        try:
            out = self.mu(list(self.all_params.values())).flatten()
        except ValueError:
            # print("Exception at parameters:",self.params.T[0])
            out = np.array([1e30]*self.n_data)
        
        state['model'] = out
    #########################################

    #########################################
    def get_model(self):
        return self.current_state['model']
    #########################################

    #########################################
    def get_allow_agnostic(self):
        return True
    #########################################

#########################################
