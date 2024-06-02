import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory
   
#########################################
class PowSineTheory(Theory):
    X = None
    #########################################
    def initialize(self):
        pass
    #########################################

    #########################################
    def calculate(self,state, want_derived=False, **param_dict):
        keys = list(param_dict.keys())
        output = param_dict[keys[0]]*np.fabs(self.X)**param_dict[keys[1]]
        output *= np.sin(param_dict[keys[2]]+self.X)**2
        state['model'] = output
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


    
#########################################
class GaussMixTheory(Theory):
    X = None
    n_comp = 1
    #########################################
    def initialize(self):
        print('Model is {0:d}-component Gaussian mixture'.format(self.n_comp))
    #########################################

    #########################################
    def calculate(self,state, want_derived=False, **param_dict):
        keys = list(param_dict.keys())        
        out = np.zeros_like(self.X)
        for c in range(self.n_comp):
            amp,mu,lnsig2 = param_dict[keys[3*c]],param_dict[keys[3*c+1]],param_dict[keys[3*c+2]]
            out += amp*np.exp(-0.5*(self.X-mu)**2/np.exp(lnsig2))
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
