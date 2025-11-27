import numpy as np
import sys,os
from pathlib import Path

from reion_uvlf_funcs import model_and_data

sys.path.append('../../code/')
from mountaineer import Model


#############################################
class UVLF_Reionization(Model):
    def __init__(self,n_params=9,fixed_params={},prior=None):
        # expect fixed_params to be dictionary with keys as subset of ['a0','a1',...] and values of these params
        self.fixed_params = fixed_params
        n_tot = n_params + len(self.fixed_params.keys())
        if n_tot != 9:
            raise Exception('UVLF_Reionization works with 9 params, but {0:d} specified.'.format(n_tot))
            
        self.all_params = {'a{0:d}'.format(p):0.0 for p in range(9)} # re-usable dictionary
        self.varied_params = list(self.all_params.keys())
        for par in self.fixed_params.keys():
            self.varied_params.remove(par)
            self.all_params[par] = self.fixed_params[par]
        # now self.varied_params is list of names of varied parameters and self.all_params is updated with fixed_params values
        print('varied params: ',self.varied_params)
        
        self.prior = prior # if not None, expect dict with keys being subset of self.varied_params at this point,
                           # and values [mean,std] of prior for that key, having accounted for log10
        if self.prior is not None:
            prior_mean = []
            prior_invsig2 = []
            for key in self.varied_params:
                if key in self.prior.keys():
                    prior_mean.append(self.prior[key][0])
                    prior_invsig2.append(1/(self.prior[key][1]**2 + 1e-30))
                else:
                    prior_mean.append(0.0)
                    prior_invsig2.append(0.0)
            prior_mean = np.array(prior_mean)
            prior_invsig2 = np.array(prior_invsig2)
        else:
            prior_mean = np.zeros(n_params)
            prior_invsig2 = np.zeros(n_params)
        
        Model.__init__(self,n_params=n_params,prior_invsig2=prior_invsig2,prior_mean=prior_mean)
        self.dlntheta = 1e-2*np.ones(n_params) # should be much smaller than typical step-size in each direction
        data,dum2,dum3 = model_and_data(-0.19, -0.92, 13.0, 2.1, 4.97, 0.34, -0.97, -0.41,9.32)
        self.n_data = data.size

    def calc_model(self,X):
        # lsum, ldiff, l2, l3, asum, adiff,  log10_fesc10, alpha_esc, log10Mcrit = self.params.T[0]
        # try:
        #     out,dummy1,dummy2 = model_and_data(lsum, ldiff, l2, l3, asum, adiff,  log10_fesc10, alpha_esc, log10Mcrit)
        p_this = self.params.T[0].copy()
        p = 0
        for key in self.varied_params:
            self.all_params[key] = p_this[p]
            p += 1
        a0,a1,a2,a3,a4,a5,a6,a7,a8 = self.all_params.values()
        try:
            out,dummy1,dummy2 = model_and_data(a0,a1,a2,a3,a4,a5,a6,a7,a8)
        except ValueError:
            # print("Exception at parameters:",self.params.T[0])
            out = np.array([1e30]*self.n_data)
            
        self.model_fid = out.copy()
        return self.rv(out[X[0]]) # X.shape = (1,n_samp)
    
    def calc_dmdtheta(self):
        # self.X,self.model_fid will be available for the data set
        dmdtheta = np.zeros((self.n_params,self.X.shape[1])) # (n_params,n_samp)
        Dtheta = np.fabs(self.params.T[0])*self.dlntheta # Dtheta
        switcher = np.ones(self.n_params)
        u = np.random.rand(self.n_params)
        switcher[u < 0.5] = -1.0
        for p in range(self.n_params):
            params_vary = self.params.T[0].copy()
            params_vary[p] += switcher[p]*Dtheta[p] # theta +- Dtheta
            # lsum, ldiff, l2, l3, asum, adiff,  log10_fesc10, alpha_esc, log10Mcrit = params_vary
            # try:
            #     model_vary,dum1,dum2 = model_and_data(lsum, ldiff, l2, l3, asum, adiff,  log10_fesc10, alpha_esc, log10Mcrit) 
            pv = 0
            for key in self.all_params.keys():
                if key in self.varied_params:
                    self.all_params[key] = params_vary[pv]
                    pv += 1
                else:
                    self.all_params[key] = self.fixed_params[key]
            a0,a1,a2,a3,a4,a5,a6,a7,a8 = self.all_params.values()
            try:
                model_vary,dummy1,dummy2 = model_and_data(a0,a1,a2,a3,a4,a5,a6,a7,a8)
                # M(theta +- Dtheta)
            except ValueError:
                # print("Exception at parameters:",params_p)
                model_vary = np.array([1e30]*self.n_data)
            dmdtheta[p] = switcher[p]*(model_vary[self.X[0]] - self.model_fid[self.X[0]]) 
            # +- [ Model(theta +- Dtheta) - Model(theta) ]
            # = +- Model(theta +- Dtheta) -+ Model(theta)
            # ... = (+): Model(theta + Dtheta) - Model(theta)
            # ... = (-): Model(theta) - Model(theta - Dtheta)

        dmdtheta = dmdtheta.T
        dmdtheta /= (Dtheta + 1e-15)
        dmdtheta = dmdtheta.T
        
        return dmdtheta
#############################################
