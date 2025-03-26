import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory

import sys
from paths import ML_Path
sys.path.append(ML_Path)
from mlstats import Chi2Like


#########################################
class EmulLike(Likelihood):
    #########################################
    def initialize(self):
        pass
    #########################################

    #########################################
    def get_requirements(self):
        """ Theory code should return model array. """
        return {'model': None}
    #########################################

    #########################################
    def logp(self,**params_values_dict):
        chi2 = self.provider.get_model()
        return -0.5*chi2
    #########################################

#########################################

   
#########################################
class NNTheory(Theory):
    net = None
    keys = [] # list of parameter names
    #########################################
    def initialize(self):
        pass
    #########################################

    #########################################
    def calculate(self,state, want_derived=False, **param_dict):
        params = self.net.cv([param_dict[key] for key in self.keys])
        chi2 = self.net.predict(params)  
        state['model'] = chi2
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
class GPRTheory(Theory):
    gprt = None
    interpolator = None
    keys = [] # list of parameter names
    #########################################
    def initialize(self):
        pass
    #########################################

    #########################################
    def calculate(self,state, want_derived=False, **param_dict):
        params = np.array([param_dict[key] for key in self.keys])
        chi2 = self.gprt.predict(params,self.interpolator)  
        state['model'] = chi2
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
