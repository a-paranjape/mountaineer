import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory

import sys
sys.path.append('../')
from mountaineer import Chi2

#########################################
class Chi2Like(Likelihood):
    X = None
    Y = None
    invcov_mat = None
    #########################################
    def initialize(self):
        self.loss_params = {'Y_full':self.Y,'invcov_mat':self.invcov_mat}
        self.loss = Chi2(params=self.loss_params)
    #########################################

    #########################################
    def get_requirements(self):
        """ Theory code should return model array. """
        return {'model': None}
    #########################################

    #########################################
    def logp(self,**params_values_dict):
        model = self.provider.get_model()
        chi2 = self.loss.forward(model) # ensure model has shape (1,n_samp)
        return -0.5*chi2
    #########################################

#########################################


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
