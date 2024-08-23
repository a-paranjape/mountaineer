import numpy as np
import scipy.linalg as linalg
import sys

from paths import ML_Path
sys.path.append(ML_Path)
from utilities import Utilities
from mllib import MLUtilities
from mlmodules import Module
import copy,pickle

import matplotlib.pyplot as plt
import gc

class Chi2(Module,MLUtilities):
    def __init__(self,params={}):
        self.Y_full = params.get('Y_full',None)
        if self.Y_full is None:
            raise ValueError("Need to supply targets on full sample (training+validation) as key 'Y_full' in Chi2.")        
        
        cov_mat = params.get('cov_mat',None)
        self.cov_mat = np.eye(self.Y_full.shape[1]) if cov_mat is None else cov_mat
        if self.cov_mat.shape != (self.Y_full.shape[1],self.Y_full.shape[1]):
            raise ValueError('Incompatible cov_mat in Chi2.')
        if np.any(linalg.eigvals(self.cov_mat) <= 0.0):
            raise ValueError('Non-positive definite covariance matrix detected')
        # this is full cov mat, to be passed only once by user in initial parameter dictionary

        self.subset = params.get('subset',np.s_[:]) # slice or array of indices, used to extract (shuffled) training or validation elems.
        self.Y_sub = self.Y_full[:,self.subset].copy()
        self.n_sub = self.Y_sub.shape[1] # n_sub = n_samp for this data set

        self.cov_mat_sub = self.cov_mat[:,self.subset][self.subset,:].copy()
        self.L_sub = linalg.cholesky(self.cov_mat_sub,lower=True) # so C = L L^T
        
        self.slice_b = params.get('slice_b',np.s_[:]) # batch slice, indexed on self.subset
        self.batch_mask = np.zeros(self.n_sub)
        self.batch_mask[self.slice_b] = 1.0 # batch selection mask
        
        # loss = sum_{i,j} res_i Cinv_{ij} res_j
        #      = sum_{i=1}^{nsamp} res_i sum_{j=1}^{nsamp} Cinv_{ij} res_j
        #      = sum_{i=1}^{nsamp} Loss_i
        # where
        # Loss_i = res_i sum_{j=1}^{nsamp} Cinv_{ij} res_j = res_i Lvec_i
        #
        # note res_i = Ypred_i - Y_i = m_i - Y_i
        #
        # d batch_loss / d m_k = sum_{i in batch} d(res_i Lvec_i) / d m_k
        #                      = sum_{i in batch} Lvec_i d res_i / d m_k + res_i d Lvec_i / d m_k
        #                      = sum_{i in batch} Lvec_i delta_{ik} + res_i d/dm_k sum_j Cinv_{ij} res_j
        #                      = (Lvec_k if k in batch else 0) + res_i sum_j Cinv_{ij} d res_j / d m_k 
        #                      = (Lvec_k if k in batch else 0) + res_i sum_j Cinv_{ij} delta_{jk} 
        #                      = (Lvec_k if k in batch else 0) + sum_{i in batch} res_i Cinv_{ik}
    
    def forward(self,Ypred_sub):
        resid_sub = Ypred_sub - self.Y_sub # (1,n_samp)
        self.resid_b = resid_sub*self.batch_mask # (1,n_samp) with non-zero only in batch
        z = linalg.cho_solve((self.L_sub,True),resid_sub[0],check_finite=False) # solves (L L^T) z = resid or z = C^-1 residual, shape (nsamp,)
        self.Loss_vec = self.cv(z*self.batch_mask) # (n_samp,1) with non-zero only in batch
        Loss = np.dot(self.resid_b,self.Loss_vec) # (1,1)
        return Loss[0,0] # scalar

    def backward(self):
        dLdm = self.Loss_vec.T # (1,n_samp)
        z = self.rv(linalg.cho_solve((self.L_sub,True),self.resid_b[0],check_finite=False))
        # solves (L L^T) z^T = resid or z = (C^-1 resid)^T, shape (1,n_samp)
        dLdm += z
        return dLdm


class Walker(Module,MLUtilities,Utilities):
    def __init__(self,data_pack={}):
        """ Class to implement gradient descent for single walker.
            data_pack should be dictionary with a subset of following keys:
            -- 'X': array of X [control variable] (1,n_samp)
            -- 'Y': array of Y [target] (1,n_samp)
            -- 'val_frac': float between 0..1, fraction of data to use for validation in Walker.train()
            -- 'model': Instance of Model containing forward (prediction),
                                   backward (gradient update) and sgd_step (with adam support) methods.
            -- 'params_init': starting array of parameters (n_param,1), compatible with model
            -- 'loss': Loss function module containing forward (prediction) 
                                and backward (gradient update) methods. Should be compatible with X,Y.
            -- 'seed': int, random number seed.
            -- 'file_stem': str, common stem for generating filenames for saving 
                                       (should include full path).
            -- 'walks_exist': boolean. If True, data files will not be overwritten and training will not be allowed. Default False. 
            -- 'verbose': boolean, whether or not to print output (default True).
            -- 'logfile': None or str, file into which to print output (default None, print to stdout)

            Provides train method that uses sgd to train on given data set.
        """
        Utilities.__init__(self)
        self.data_pack = data_pack
        self.X = data_pack.get('X',None)
        self.Y = data_pack.get('Y',None)
        self.val_frac = data_pack.get('val_frac',0.2) # fraction of input data to use for validation
        self.model = copy.deepcopy(data_pack.get('model',None))
        self.params_init = data_pack.get('params_init',None)
        
        self.loss_module = data_pack.get('loss',Chi2) 
        self.seed = data_pack.get('seed',None)
        self.file_stem = data_pack.get('file_stem','walk')
        self.walks_exist = data_pack.get('walks_exist',False)
        self.verbose = data_pack.get('verbose',True)
        self.logfile = data_pack.get('logfile',None)
        
        self.rng = np.random.RandomState(self.seed)
        
        if (self.X is None) | (self.Y is None):
            raise ValueError("Data X,Y must be specified in Walker().")
            
        if self.X.shape != self.Y.shape:
            raise ValueError("Incompatible X,Y in Walker(). Must have same shape.")
        
        if self.X.shape[0] != 1:
            raise TypeError("Only 1-d data supported in Walker().")
            
        if self.params_init.shape != (self.model.n_params,1):
            raise TypeError("Incompatible initial parameters in Walker().")
            
        self.out_file = self.file_stem + '_loss_params.txt'
        if not self.walks_exist:
            with open(self.out_file,'w') as f:
                f.write('# loss | a0 ... a{0:d}\n'.format(self.model.n_params))
        
        self.n_samp = self.X.shape[1]
        if self.walks_exist:
            walk = self.load()
            self.model.params = self.cv(walk[1:,-1])
        else:
            self.model.params = self.params_init
            
        self.n_val = np.rint(self.val_frac*self.n_samp).astype(int)
        self.n_samp -= self.n_val        

        self.ind_val = self.rng.choice(self.Y.shape[1],size=self.n_val,replace=False)
        self.ind_train = np.delete(np.arange(self.Y.shape[1]),self.ind_val) 
        # Note ind_train is ordered although ind_val is randomised

        self.X_train = self.X[:,self.ind_train].copy()
        self.Y_train = self.Y[:,self.ind_train].copy()

        self.X_val = self.X[:,self.ind_val].copy()
        self.Y_val = self.Y[:,self.ind_val].copy()
        
    def train(self,params={}):
        """ Main routine for training.
            Expect X.shape = (n0,n_samp) = Y.shape
        """
        if self.walks_exist:
            raise Exception('Walks exist! No training allowed.')
        max_epoch = params.get('max_epoch',20)
        lrate = params.get('lrate',0.1)
        mb_count = params.get('mb_count',int(np.sqrt(self.n_samp)))
        loss_params = copy.deepcopy(params.get('loss_params',{})) # dictionary for initialising loss module mini-batch-wise
        loss_params['Y_full'] = self.Y.copy() # full target data; loaded independent of loss type

        check_after = params.get('check_after',10)
        check_after = np.max([check_after,2])
        
        if self.verbose:
            self.print_this("... training",self.logfile)
            
        if (mb_count > self.n_samp) | (mb_count < 1):
            if self.verbose:
                self.print_this("Incompatible mb_count in Walker.sgd(). Setting to n_samp (standard SGD).",
                                self.logfile)
            mb_count = self.n_samp
        if (mb_count < self.n_samp) & (mb_count > np.sqrt(self.n_samp)):
            if self.verbose:
                print_str = "Large mb_count might lead to uneven mini-batch sizes in Walker.sgd()."
                print_str += " Setting to int(sqrt(n_samp))."
                self.print_this(print_str,self.logfile)
            mb_count = int(np.sqrt(self.n_samp))
            
        mb_size = self.n_samp // mb_count

        self.epochs = np.arange(max_epoch)+1.0
        self.val_loss = np.zeros(max_epoch)
        ind_shuff = np.arange(self.n_samp)
        for t in range(max_epoch):
            self.rng.shuffle(ind_shuff)
            X_train_shuff = self.X_train[:,ind_shuff].copy() 
            for b in range(mb_count):
                self.save(loss_params=loss_params) # will set keys to *full* data and save total loss
                
                loss_params['subset'] = self.ind_train[ind_shuff].copy()                
                sl = np.s_[b*mb_size:(b+1)*mb_size] if b < mb_count-1 else np.s_[b*mb_size:]
                
                Ypred = self.model.forward(X_train_shuff) # update activations
                loss_params['slice_b'] = sl # setting this ensures loss calculated for mini-batch
                self.loss = self.loss_module(params=loss_params)
                batch_loss = self.loss.forward(Ypred) # calculate current batch loss, update self.loss
                dLdm = self.loss.backward() # loss.backward returns d(loss)/d(model)

                self.model.backward(dLdm) # update gradients
                self.model.sgd_step(t,lrate) # gradient descent update
                
            # validation check
            loss_params['subset'] = self.ind_val.copy()
            loss_params['slice_b'] = np.s_[:]
            self.loss = self.loss_module(params=loss_params)
            Ypred_val = self.model.forward(self.X_val) # update activations. prediction for validation data
            self.val_loss[t] = self.loss.forward(Ypred_val) # calculate validation loss, update self.loss
            if t > check_after:
                x = np.arange(t-check_after,t+1)
                y = self.val_loss[x].copy()
                # xbar = np.mean(x)
                # slope = (np.mean(x*y)-xbar*np.mean(y))/(np.mean(x**2) - xbar**2 + 1e-15) # best fit slope
                if np.mean(x*y)-np.mean(x)*np.mean(y) > 0.0: # check for positive sign of best fit slope
                    if self.verbose:
                        self.print_this('',self.logfile)
                    break
            # if t > check_after:
            #     chk_half = (self.val_loss[t] > self.val_loss[t-check_after//2])
            #     chk = (self.val_loss[t-check_after//2] > self.val_loss[t-check_after])
            #     if chk_half & chk:
            #         if self.verbose:
            #             self.print_this('',self.logfile)
            #         break
                
            if self.verbose:
                self.status_bar(t,max_epoch)
                
        self.save(save_setup=True,loss_params=loss_params)
        
        if self.verbose:
            self.print_this("... ... done",self.logfile)
            
        return

    def predict(self,X):
        """ Predict targets for given data set. """
        if X.shape[0] != 1:
            raise TypeError("Incompatible data in Walker.predict(). Expected 1, got {0:d}".format(X.shape[0]))
        # update and predict._b        
        return self.model.forward(X)

    def save(self,save_setup=False,loss_params={}):
        """ Save current total loss + params and (optionally) data_pack to file(s). """
        lp = copy.deepcopy(loss_params)
        lp['subset'] = np.s_[:]
        lp['slice_b'] = np.s_[:]
        self.loss = self.loss_module(params=lp)

        Ypred = self.model.forward(self.X)
        loss = self.loss.forward(Ypred)
        
        seq = [loss] + list(self.model.params.T[0])
        self.write_to_file(self.out_file,seq)

        if save_setup:
            with open(self.file_stem + '.pkl', 'wb') as f:
                pickle.dump(self.data_pack,f)
            
        return    

    # to be called after generating instance of Walker() with correct setup data_pack
    def load(self):
        """ Load loss values and data_pack from file. """
        out = np.loadtxt(self.out_file).T
        with open(self.file_stem + '.pkl', 'rb') as f:
            self.data_pack = pickle.load(f)
        return out


class Model(Module,MLUtilities):
    def __init__(self,n_params=None,adam=True,B1_adam=0.9,B2_adam=0.999,eps_adam=1e-8):
        self.n_params = n_params
        if self.n_params is None:
            raise ValueError('n_params must be integer in Model().')

        self.params = np.zeros((self.n_params,1))
        self.dLdtheta = np.zeros_like(self.params)
        
        # adam support
        self.adam = adam
        self.B1_adam = B1_adam
        self.B2_adam = B2_adam
        self.eps_adam = eps_adam
        if self.adam:
            self.M = np.zeros_like(self.params)
            self.V = np.zeros_like(self.params)
    
    def forward(self,X):
        self.X = X
        return self.calc_model(X)
    
    def backward(self,dLdm):
        # expect dLdm.shape = (1,n_samp)
        dmdtheta = self.calc_dmdtheta() # (n_params,n_samp)

        self.dLdtheta = np.dot(dmdtheta,dLdm.T) # (n_params,1)
        if self.adam:
            self.M = self.B1_adam*self.M + (1-self.B1_adam)*self.dLdtheta
            self.V = self.B2_adam*self.V + (1-self.B2_adam)*self.dLdtheta**2
        return

    def calc_model(self,X):
        prnt_strng = "User must specify calc_model method of Model instance."
        prnt_strng += " Output must be model prediction of shape X.shape = (1,n_samp)."
        raise NotImplementedError(prnt_strng)

    def calc_dmdtheta(self):
        # expect dLdm.shape = (1,n_samp)
        prnt_strng = "User must specify calc_dmdtheta method of Model instance." 
        prnt_strng += " Method must return dmdtheta (n_params,n_samp) using input dLdm (1,n_samp)."
        raise NotImplementedError(prnt_strng)
    
    def sgd_step(self,t,lrate):
        if self.adam:
            corr_B1 = 1-self.B1_adam**(1+t) 
            corr_B2 = 1-self.B2_adam**(1+t)
            dtheta = self.M/corr_B1/np.sqrt(self.V/corr_B2 + self.eps_adam)
        else:
            dtheta = self.dLdtheta
        self.params = self.params - lrate*dtheta
        
        return 


class Mountaineer(Module,MLUtilities,Utilities):
    """ Main routines for initialising walkers and climbing. """
    ###########################################
    def __init__(self,data_pack={}):
        """ Main class to initialise and train all walkers.

            data_pack should be dictionary with a subset of following keys:

            ** needed for self instance **
            -- 'N_evals_max': int, maximum number of model evaluations allowed. Default 100.
            -- 'survey_frac': float, fraction of N_evals_max to use for initial survey. Set to zero to skip survey step. Default 0.1.
            -- 'file_stem': str, common stem for generating filenames for saving 
                                       (should include full path).
            -- 'model': sub-class of Model with user-defined calc_model() and calc_dmdtheta() methods.
            -- 'n_params': int, number of parameters in model.
            -- 'param_mins','param_maxs': array-likes of floats (n_params,) - guesses for minimum and maximum values for parameters 
            
            ** passed to Walker instances **
            -- 'X': array of X [control variable] (1,n_samp)
            -- 'Y': array of Y [target] (1,n_samp)
            -- 'val_frac': float between 0..1, fraction of data to use for validation in Walker.train(). Default 0.2.
            -- 'loss': Loss function module containing forward (prediction) 
                                and backward (gradient update) methods. Should be compatible with X,Y.
            -- 'seed': int or None (default), random number seed.
            -- 'walks_exist': boolean. If True, data files will not be overwritten and training will not be allowed. Default False. 
            -- 'verbose': boolean, whether of not to print output (default True).
            -- 'logfile': None or str, file into which to print output (default None, print to stdout)

            ** passed to Walker.train() calls **
            -- 'loss_params': dictionary with common keys to be used by all Walker instances. 
                              E.g. for Chi2 this would be 'cov_mat' containing covariance matrix of *full* data set.

            Methods:
            -- climb: trains all requested walkers on given data set, starting at Latin hypercube of initial locations.
            -- gather: gathers outputs from all walkers and stores in single file.
            -- visualize: makes simple plots of walker outputs
        """
        Utilities.__init__(self)
        self.N_evals_max = data_pack.get('N_evals_max',100)
        self.survey_frac = data_pack.get('survey_frac',0.1)
        self.file_stem = data_pack.get('file_stem','walk')
        self.walks_file = self.file_stem + '_all.txt'
        self.range_file = self.file_stem + '_ranges.txt'
        self.Model = data_pack.get('model',None)
        self.n_params = data_pack.get('n_params',None)
        self.param_mins = data_pack.get('param_mins',None)
        self.param_maxs = data_pack.get('param_maxs',None)
        ####
        # will be set by self.create_survey()
        self.survey_params = None
        self.survey_lhc_layers = None
        self.survey_loss = None
        self.survey_dLdtheta = None
        ####
        # will be set by self.distribute()
        self.N_walker = None
        self.walker_layers = None
        self.lrates = None
        ####
        # give this to user-control if needed
        # no. of surveys to average over when updating param ranges
        self.n_iter_survey = 5

        # internally set
        self.N_survey = int(self.survey_frac*self.N_evals_max)
        self.N_survey_tot = self.N_survey*self.n_iter_survey # will be updated to actual number in self.survey()
        self.N_evals_max_walk = self.N_evals_max - self.N_survey_tot
        self.N_walker = 3*self.n_params 
        self.N_survey_lhc_layers = (self.N_survey // 2) + 1
        # min,max values of lrate
        self.lrate_min = 0.15
        self.lrate_max = 0.15 # will be changed by self.set_lrates()
        self.lrate_largest_max = 0.25
                
        # array of adam parameter values for different walkers.
        # will be set by self.set_adams() called by self.distribute()
        self.B1_adams = None
        self.B2_adams = None
        # min,max values
        self.B1_adam_min,self.B1_adam_max = 0.75,0.9 # (default 0.9) 0.75,0.9
        self.B2_adam_min,self.B2_adam_max = 0.75,(1-1e-6) # (default 0.999) 0.75,1-1e-6

        self.X = data_pack.get('X',None)
        self.Y = data_pack.get('Y',None)
        
        self.check_init()
        
        self.param_mins = np.array(self.param_mins)
        self.param_maxs = np.array(self.param_maxs)
        
        self.model_inst = self.Model(n_params=self.n_params)#,adam=self.adam,
                                     # B1_adam=self.B1_adam,B2_adam=self.B2_adam,eps_adam=self.eps_adam)
        
        self.val_frac = data_pack.get('val_frac',0.2) # fraction of input data to use for validation
        
        self.loss_module = data_pack.get('loss',Chi2) 
        self.seed = data_pack.get('seed',None)
        self.walks_exist = data_pack.get('walks_exist',False)
        self.verbose = data_pack.get('verbose',True)
        self.logfile = data_pack.get('logfile',None)

        self.rng = np.random.RandomState(self.seed) # only used in self.adjust_survey for shuffling param directions
        
        if self.verbose:
            self.print_this('Mountaineer to explore loss land-scape!',self.logfile)
            
        self.loss_params = copy.deepcopy(data_pack.get('loss_params',{}))        
        self.distributed = 0 # will be set to 1 upon successful call to self.distribute().
        self.mb_count = int(np.sqrt(self.X.shape[1])) # passed to Walker after distribution

        #######################
        # Following strategies for survey adjustment were considered and discarded:
        # - loss difference threshold: won't work for ring-like likelihoods
        # - delete largest loss points with threshold on survivor number: unstable and non-uniform
        # Currently using method that 'peels away' survey LHC layers outside-in while average div(grad loss) stays positive.
        
        if self.verbose:
            self.print_this('... initialization done',self.logfile)
    ###########################################
        
    ###########################################
    def check_init(self):
        if self.Model is None:
            raise ValueError("Need to specify valid Model sub-class in Mountaineer.")

        if self.n_params is None:
            raise ValueError("Need to specify valid number of model parameters in Mountaineer.")

        if self.param_mins is None:
            raise ValueError("Need to specify valid array of initial param_mins in Mountaineer.")
        if self.param_maxs is None:
            raise ValueError("Need to specify valid array of initial param_maxs in Mountaineer.")

        if len(self.param_mins) != self.n_params:
            raise ValueError('Incompatible param_mins and n_params in Mountaineer.')

        if len(self.param_maxs) != self.n_params:
            raise ValueError('Incompatible param_maxs and n_params in Mountaineer.')

        if self.X is None:
            raise ValueError("Need to specify valid data X in Mountaineer.")

        if self.Y is None:
            raise ValueError("Need to specify valid data Y in Mountaineer.")
        
        if self.N_evals_max_walk <= self.N_walker:        
            raise Exception('{0:d} evals is not enough for exploration.'.format(self.N_evals_max_walk))
        
        return
    ###########################################
    
    ###########################################
    def explore(self):
        """ User-friendly wrapper to perform all tasks:
            * use small fraction of N_evals_max to update param_mins,param_maxs
            * judiciously distribute remaining evals amongst walkers and decide their training parameters
            * train all walkers and store outputs
            * gather all outputs
            Returns: 
            -- walks: list of arrays, each element is output of Walker.load() [can be fed to visualize().]
        """
        if not self.walks_exist:
            # initial survey
            self.survey()
            # distribute resources and set training parameters
            self.distribute()
            if self.distributed == 1:
                # train all walkers
                self.climb()
            else:
                if self.verbose:
                    self.print_this('Not enough walkers. Exploration failed.',self.logfile)
                return None
        else:
            if self.verbose:
                self.print_this('Walks exist. No survey or climbing.',self.logfile)

        # consolidate walker outputs
        walks = self.gather()
        
        return walks
    ###########################################

    ###########################################
    def survey(self):
        """ Use small fraction survey_frac of N_evals_max to do initial survey and update param_mins,param_maxs if needed. """
        if self.walks_exist:
            raise Exception('Walks exist! No surveying allowed.')

        if self.N_survey <= 0:
            if self.verbose:
                self.print_this('Skipping survey...',self.logfile)
            return

        self.param_maxs_old = copy.deepcopy(self.param_maxs)
        self.param_mins_old = copy.deepcopy(self.param_mins)
            
        model_survey = copy.deepcopy(self.model_inst) # AVOID COPYING IF POSSIBLE

        if self.verbose:
            self.print_this('Surveying using {0:d} locations {1:d} times...'.format(self.N_survey,self.n_iter_survey),self.logfile)
        pmaxs = np.zeros((self.n_iter_survey,self.n_params))
        pmins = np.zeros((self.n_iter_survey,self.n_params))
        sparam = []
        sloss = []
        for i in range(self.n_iter_survey):
            if self.verbose:
                self.print_this('... iteration {0:d}'.format(i+1),self.logfile)
            # below two will create and share self.survey_{params,lhc_layers,loss,dLdtheta} arrays, of which params and loss will be stored
            self.create_survey(model_survey)
            self.adjust_survey()
            pmaxs[i] = self.param_maxs
            pmins[i] = self.param_mins
            sparam.append(self.survey_params)
            sloss.append(self.survey_loss)
            # reset ranges for next iteration
            self.param_maxs = copy.deepcopy(self.param_maxs_old)
            self.param_mins = copy.deepcopy(self.param_mins_old)

        self.param_maxs = np.median(pmaxs,axis=0)
        self.param_mins = np.median(pmins,axis=0)
        
        self.survey_params = np.zeros((self.N_survey_tot,self.n_params))
        self.survey_loss = np.zeros(self.N_survey_tot)
        for i in range(self.n_iter_survey):
            self.survey_params[i*self.N_survey:(i+1)*self.N_survey,:] = sparam[i]
            self.survey_loss[i*self.N_survey:(i+1)*self.N_survey] = sloss[i]

        self.survey_params,idx = np.unique(self.survey_params,axis=0,return_index=True)
        self.survey_loss = self.survey_loss[idx]
        if self.survey_loss.size != self.N_survey_tot:
            if self.verbose:
                self.print_this('... eliminated {0:d} repeated param vectors'.format(self.N_survey_tot-self.survey_loss.size),self.logfile)                
        self.N_survey_tot = self.survey_loss.size
        
        if self.verbose:
            prnt_str = '... old param_mins  = ['+','.join(['{0:.2e}'.format(p) for p in self.param_mins_old]) +']\n'
            prnt_str += '... ... modified to = ['+','.join(['{0:.2e}'.format(p) for p in self.param_mins]) +']\n'
            prnt_str += '... old param_maxs  = ['+','.join(['{0:.2e}'.format(p) for p in self.param_maxs_old]) +']\n'
            prnt_str += '... ... modified to = ['+','.join(['{0:.2e}'.format(p) for p in self.param_maxs]) +']'
            self.print_this(prnt_str,self.logfile)
            
        return
    ###########################################

    ###########################################
    def create_survey(self,model_survey):
        """ Set up survey. """
        if self.verbose:
            self.print_this('... creating survey',self.logfile)
        survey_params,survey_lhc_layers = self.gen_latin_hypercube(Nsamp=self.N_survey,dim=self.n_params,return_layers=True,
                                                                   param_mins=self.param_mins,param_maxs=self.param_maxs)
        # survey_params has shape (N_survey,n_params)
        #        layers ..  ..    (Nsurvey,)
        
        if survey_lhc_layers.max() + 1 > self.N_survey_lhc_layers:
            raise Exception('Too many layers in LHC! Should be {0:d}, found at least {1:d}'
                            .format(self.N_survey_lhc_layers,survey_lhc_layers.max()+1))

        if self.verbose:
            self.print_this('... ... evaluating loss values and gradients',self.logfile)
        lp = copy.deepcopy(self.loss_params)
        lp['Y_full'] = self.Y.copy() # since Walker is not invoked
        loss = self.loss_module(params=lp)
        survey_loss = np.zeros(self.N_survey)
        survey_dLdtheta = np.zeros((self.N_survey,self.n_params))
        for s in range(self.N_survey):
            model_survey.params = survey_params[s:s+1,:].T # shape (n_params,1)
            # calculate total loss at survey parameters
            Ypred = model_survey.forward(self.X) 
            survey_loss[s] = loss.forward(Ypred)
            dLdm = loss.backward()
            model_survey.backward(dLdm) # adam variables (M,V) updated but not used since no sgd_step invoked
            survey_dLdtheta[s] = model_survey.dLdtheta[:,0]
            # write this somewhere so as to not lose evaluations !
            if self.verbose:
                self.status_bar(s,self.N_survey)
                
        # NaN is my enemy
        if self.verbose:
            self.print_this('... ... excluding NaNs',self.logfile)
        cond_notnan = ~np.isnan(survey_loss)
        survey_loss = survey_loss[cond_notnan]
        survey_params = survey_params[cond_notnan]
        survey_lhc_layers = survey_lhc_layers[cond_notnan]
        survey_dLdtheta = survey_dLdtheta[cond_notnan]
        
        if self.verbose:
            self.print_this('... ... kept {0:d} of {1:d} surveyed points'.format(survey_loss.size,self.N_survey),self.logfile)

        self.survey_params = survey_params.copy()
        self.survey_lhc_layers = survey_lhc_layers.copy()
        self.survey_loss = survey_loss.copy()
        self.survey_dLdtheta = survey_dLdtheta.copy()
        
        return 
    ###########################################
    
    ###########################################
    def adjust_survey(self):
        """ Update param_mins,param_maxs by stepping inwards through LHC layers so long as loss gradients remain 'inward-pointing'. """
        if (self.survey_params is None):
            raise Exception("adjust_survey() can only be called after create_survey().")
        
        if self.verbose:
            self.print_this('... adjusting survey',self.logfile)
            
        Dp = (self.param_maxs - self.param_mins)/self.N_survey
        # Dp[d] = division size in direction d
                
        if self.verbose:
            self.print_this('... ... looping through layers',self.logfile)
            
        # area elements are prod Dp over all but direction in question
        dArea = np.zeros(self.n_params)
        for d in range(self.n_params):
            dArea[d] = np.prod(Dp[np.delete(np.arange(self.n_params),d)])
                
        # average divergence of loss
        for l in range(self.N_survey_lhc_layers):
            # locations of max and min values in layer along each direction
            max_layer = self.param_maxs - l*Dp
            min_layer = self.param_mins + l*Dp
            # indices of survey points in layer
            points = np.where(self.survey_lhc_layers == l)[0]
            # points.size guaranteed positive due to LHC construction
            if points.size:
                # condition to proceed to next layer: vol_avg div(grad loss) > 0, over vol enclosed by layer
                div_grad_loss = 0.0 # collect contribution from each point on layer.
                for p in range(points.size):
                    max_dims = np.where(np.fabs(self.survey_params[points[p]] - max_layer) < 0.1*Dp)[0] 
                    min_dims = np.where(np.fabs(self.survey_params[points[p]] - min_layer) < 0.1*Dp)[0]
                    nhat = np.zeros(self.n_params)
                    nhat[max_dims] = 1.0
                    nhat[min_dims] = -1.0
                    nhat /= np.sqrt(np.sum(nhat**2))
                    div_grad_loss += np.dot(self.survey_dLdtheta[points[p]],nhat*dArea)
                # note loss ~ -ln(likelihood). we want likelihood grads pointed inwards, hence loss grads pointed outwards.
                if div_grad_loss < 0.0:
                    if self.verbose:
                        self.print_this('... ... avg div(grad loss) negative at layer {0:d}; breaking.'.format(l),self.logfile)
                    break
                # else proceed to next layer
            
        l_stay = np.max([0,l]) # use last layer. can be made more conservative later.
        if l_stay != self.N_survey_lhc_layers-1:
            if self.verbose:
                self.print_this('... ... adjusting parameter ranges',self.logfile)
            self.param_maxs -= l_stay*Dp
            self.param_mins += l_stay*Dp
        else:
            # if we sailed thru then probably no change was ever needed!
            if self.verbose:
                self.print_this('... ... no adjustment needed (probably!)',self.logfile)
        
        return 
    ###########################################
    
    ###########################################
    def distribute(self):
        """ Distribute available resources among walkers and set training parameters."""                    
        if self.walks_exist:
            if self.verbose:
                self.print_this('Setting up pre-existing walks and ranges...',self.logfile)
            # expect self.range_file to be written already
            ranges = np.loadtxt(self.range_file).T
            self.param_mins = ranges[0].copy()
            self.param_maxs = ranges[1].copy()
        else:
            if self.verbose:
                self.print_this('Distributing available resources ({0:d} evals)...'.format(self.N_evals_max_walk),self.logfile)
        
        if self.verbose:
            self.print_this('... initialising {0:d} walkers'.format(self.N_walker),self.logfile)
            
        dp_walk = {'X':self.X,'Y':self.Y,'val_frac':self.val_frac,
                   'model':copy.deepcopy(self.model_inst), # copying since adam needs adjustment. ** can we avoid copying? **
                   'loss':self.loss_module,'walks_exist':self.walks_exist,
                   'seed':self.seed,'verbose':False,'logfile':self.logfile}

        self.walkers = []
        pins,self.walker_layers = self.gen_latin_hypercube(Nsamp=self.N_walker,dim=self.n_params,return_layers=True,
                                                           param_mins=self.param_mins,param_maxs=self.param_maxs)
        self.set_adams() # will set self.B1_adams,self.B2_adams
        
        for w in range(self.N_walker):
            dp_walk['file_stem'] = self.file_stem + '_w' + str(w+1)
            dp_walk['params_init'] = pins[w:w+1,:].T # values irrelevant if walks exist, but structure needed
            dp_walk['model'].B1_adam = self.B1_adams[w]
            dp_walk['model'].B2_adam = self.B2_adams[w]
            self.walkers.append(Walker(data_pack=dp_walk))

        if not self.walks_exist:
            if self.verbose:
                self.print_this('... setting up training parameters',self.logfile)

            ####################
            self.max_epoch = self.N_evals_max_walk // (self.N_walker*self.mb_count) # 30
            self.check_after = self.max_epoch // 3
            if self.check_after < 2:
                self.check_after = self.max_epoch + 1

            ####################
            self.set_lrates()

            if self.verbose:
                self.print_this('... ...   max_epoch = {0:d}'.format(self.max_epoch),self.logfile)
                self.print_this('... ...    mb_count = {0:d}'.format(self.mb_count),self.logfile)
                self.print_this('... ... check_after = {0:d}'.format(self.check_after),self.logfile)
                self.print_this('... ...      lrates = {0:.4f} to {1:.4f} (outside inwards)'.format(self.lrate_max,self.lrate_min),self.logfile)
                self.print_this('... ...  1-B1_adams = {0:.2e} to {1:.2e} (outside inwards)'.format(1-self.B1_adams.max(),1-self.B1_adams.min()),self.logfile)
                self.print_this('... ...  1-B2_adams = {0:.2e} to {1:.2e} (outside inwards)'.format(1-self.B2_adams.max(),1-self.B2_adams.min()),self.logfile)

            self.params_train = []
            for w in range(self.N_walker):
                self.params_train.append({'max_epoch':self.max_epoch,
                                          'mb_count':self.mb_count,
                                          'lrate':self.lrates[w],
                                          'loss_params':self.loss_params,
                                          'check_after':self.check_after})
        ##############        
        # DO SANITY CHECK ON VALUES OF B1_ADAM, B2_ADAM and LRATE FOR EACH WALKER
        ##############
        
        self.distributed = 1
        return 
    ###########################################

    ###########################################
    def set_adams(self):
        """ Set adam parameters B1 and B2 for each walker based on its LHC layer, 
            with outer layers having larger values of B1,B2 to make walks more ridge-directed. 
        """
        if self.N_walker is None:
            raise Exception('set_adams() can only be called within or after distribute() in Mountaineer.')
        B1_vals = np.linspace(self.B1_adam_max,self.B1_adam_min,self.walker_layers.max()+1)
        B2_vals = np.linspace(self.B2_adam_max,self.B2_adam_min,self.walker_layers.max()+1)
        self.B1_adams = np.zeros(self.N_walker)
        self.B2_adams = np.zeros(self.N_walker)
        for w in range(self.N_walker):
            self.B1_adams[w] = B1_vals[self.walker_layers[w]]
            self.B2_adams[w] = B2_vals[self.walker_layers[w]]
        return
    ###########################################

    ###########################################
    def set_lrates(self):
        """ Set learning rate for each walker based on its LHC layer, with outer layers having higher rates. """
        if self.N_walker is None:
            raise Exception('set_lrates() can only be called within or after distribute() in Mountaineer.')

        ranges = self.param_maxs_old - self.param_mins_old
        changes_pmaxs = np.fabs(self.param_maxs - self.param_maxs_old)/ranges
        changes_pmins = np.fabs(self.param_mins - self.param_mins_old)/ranges
        # above are changes in maxs,mins, relative to old ranges
        max_rel_change = np.max([changes_pmaxs.max(),changes_pmins.max()]) # largest relative change
        self.lrate_max = np.min([self.lrate_largest_max,self.lrate_max*(1+max_rel_change)])
        self.lrate_max = np.max([self.lrate_min,self.lrate_max])
        
        lrate_vals = np.linspace(self.lrate_max,self.lrate_min,self.walker_layers.max()+1)
        # linearly change from min (default 0.15) to max (minimum 0.25) from layer = walker_layers.max()+1 to 0
        self.lrates = np.zeros(self.N_walker)
        for w in range(self.N_walker):
            self.lrates[w] = lrate_vals[self.walker_layers[w]]
        return
    ###########################################    

    ###########################################
    def climb(self):
        """ Train all walkers and store outputs. Should be called after distribute. """
        if self.walks_exist:
            raise Exception('Walks exist! No climbing allowed.')
        if self.distributed == 0:
            raise Exception('Distribution not done! No climbing allowed.')
        
        if self.verbose:
            self.print_this('Climbing...',self.logfile)

        for w in range(self.N_walker):
            self.walkers[w].train(params=self.params_train[w])
            if self.verbose:
                self.status_bar(w,self.N_walker)
        
        if self.verbose:
            self.print_this('... setting final parameter ranges',self.logfile)
        pmaxs = -np.inf*np.ones(self.n_params)
        pmins = np.inf*np.ones(self.n_params)
        for w in range(self.N_walker):
            params_w = self.walkers[w].load()[1:,:] # shape (n_params,n_steps)
            cond_nan = np.isnan(np.sum(params_w,axis=0))
            params_w = params_w.T[~cond_nan] # keep only those steps where all params are non-NaN
            params_w_max = np.max(params_w,axis=0) # (n_params,)
            params_w_min = np.min(params_w,axis=0) # (n_params,)
            pmaxs = np.maximum(params_w_max,pmaxs)
            pmins = np.minimum(params_w_min,pmins)
        # now pmaxs,pmins respectively have largest,smallest non-NaN values sampled by walkers in any direction
        
        self.param_maxs = np.maximum(pmaxs,self.param_maxs)
        self.param_mins = np.minimum(pmins,self.param_mins)
        
        # don't exceed original user-specified ranges
        self.param_maxs = np.minimum(self.param_maxs_old,self.param_maxs)
        self.param_mins = np.maximum(self.param_mins_old,self.param_mins)

        if self.verbose:
            prnt_str = '... final param_mins  = ['+','.join(['{0:.2e}'.format(p) for p in self.param_mins]) +']\n'
            prnt_str += '... final param_maxs  = ['+','.join(['{0:.2e}'.format(p) for p in self.param_maxs]) +']'
            self.print_this(prnt_str,self.logfile)
            self.print_this('... writing to file: '+self.range_file,self.logfile)
        with open(self.range_file,'w') as f:
            f.write('# p_min | p_max\n')
        for p in range(self.n_params):
            self.write_to_file(self.range_file,[self.param_mins[p],self.param_maxs[p]])
        
        if self.verbose:
            self.print_this('... done',self.logfile)

        return
    ###########################################

    ###########################################
    def gather(self):
        """ Gather all walker outputs in single file. Should be called after invoking self.climb. 
            Returns: 
            -- walks: list of arrays, each element is output of Walker.load() [can be fed to visualize().]
        """
        if self.walks_exist:
            self.distribute() # this will initialise Walker instances with correct file stem
            
        walks = []
        for w in self.walkers:
            walks.append(w.load())

        if not self.walks_exist:
            if self.verbose:
                self.print_this('Writing to file: '+self.walks_file,self.logfile)            
            with open(self.walks_file,'w') as f:
                f.write('# loss | a0 ... a{0:d}\n'.format(self.n_params))
        else:
            if self.verbose:
                self.print_this('Walks exist, nothing to write.',self.logfile)            
            
        # include survey first
        steps = 1*self.N_survey_tot
        if not self.walks_exist:
            for s in range(steps):
                self.write_to_file(self.walks_file,[self.survey_loss[s]] + [self.survey_params[s,d] for d in range(self.n_params)])
        
        for w in range(self.N_walker):
            if walks[w].size > 0:
                steps_this = walks[w].shape[1]
                steps += steps_this
                if not self.walks_exist:
                    for s in range(steps_this):
                        self.write_to_file(self.walks_file,list(walks[w][:,s]))
            
        if self.verbose:
            self.print_this('... total steps taken = {0:d} (of which {1:d} were survey)'.format(steps,self.N_survey_tot),self.logfile)
            self.print_this('... done',self.logfile)            

        return walks
    ###########################################

    ###########################################
    def load(self):
        walks_file = self.file_stem + '_all.txt'
        if self.verbose:
            self.print_this('Reading from file: '+walks_file,self.logfile)
        data = np.loadtxt(walks_file).T # (n_params+1,n_steps_allwalks)

        return data
    ###########################################

    ###########################################
    def visualize(self,walks):
        """ Visualize each walk. Expect walks to be output of self.gather(). """
        if self.distributed != 1:
            raise Exception("Exploration seems to have failed. Can't visualize non-existent walks!")
        
        if self.verbose:
            self.print_this('Visualizing...',self.logfile)
            self.print_this('... setting up visualization colors',self.logfile)
        self.cols = iter(plt.cm.Spectral_r(np.linspace(0,1,self.N_walker)))

        data = self.load()
        survey = data[:,:self.N_survey_tot].copy()
        survey = survey[1:,:] # (n_params,N_survey_tot)

        cols = copy.deepcopy(self.cols)
        plt.figure(figsize=(3,3))
        plt.xlabel('x')
        plt.ylabel('y')
        errors = np.sqrt(np.diag(self.loss_params['cov_mat']))
        imin = self.Y[0].argmin()
        imax = self.Y[0].argmax()
        plt.ylim(self.Y[0][imin]-1.5*errors[imin],self.Y[0][imax]+1.5*errors[imax])
        plt.errorbar(self.X[0],self.Y[0],yerr=errors,ls='none',c='k',capsize=5,marker='o')
        for w in range(self.N_walker):
            col = next(cols)
            plt.plot(self.X[0],self.walkers[w].predict(self.X)[0],'-',color=col,lw=1)
        plt.show()

        cols = copy.deepcopy(self.cols)
        fig,ax1 = plt.subplots(figsize=(3,3))
        ax1.set_yscale('log')
        ax1.set_ylim(0.1,1e5)
        ax1.set_xlabel('step')
        ax1.set_ylabel('loss')
        ax2 = ax1.twiny()
        ax2.set_xlabel('epoch')
        for w in range(self.N_walker):
            col = next(cols)
            if np.all(np.isfinite(walks[w][0])):
                Label = 'total loss' if w==0 else None
                ax1.plot(walks[w][0],'-',color=col,lw=1,label=Label)
            if (not self.walks_exist):
                if np.all(np.isfinite(self.walkers[w].val_loss)):
                    Label = 'validation loss' if w==0 else None
                    ax2.plot(self.walkers[w].epochs,self.walkers[w].val_loss,'--',color=col,lw=1,label=Label)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        plt.show()

        for pi in range(self.n_params-1):
            for pj in range(pi+1,self.n_params):
                cols = copy.deepcopy(self.cols)
                plt.figure(figsize=(3,3))
                plt.xlabel('$a_{{{0:d}}}$'.format(pi))
                plt.ylabel('$a_{{{0:d}}}$'.format(pj))
                dpx = self.param_maxs[pi]-self.param_mins[pi]
                xmin,xmax = self.param_mins[pi]-0.05*dpx,self.param_maxs[pi]+0.05*dpx
                plt.xlim(xmin,xmax)
                dpy = self.param_maxs[pj]-self.param_mins[pj]
                ymin,ymax = self.param_mins[pj]-0.05*dpy,self.param_maxs[pj]+0.05*dpy
                plt.ylim(ymin,ymax)
                for w in range(self.N_walker):
                    col = next(cols)
                    plt.plot(walks[w][pi+1],walks[w][pj+1],ls='-',color=col,marker='o',markersize=3,lw=1)
                plt.scatter(survey[pi],survey[pj],marker='*',s=1,c='gray')
                plt.show()

        return
    ###########################################

    
    # ###########################################
    # def adjust_survey_old(self,sp,ls):
    #     """ One pass of updating param_mins,param_maxs. """
    #     survey_params = sp.copy()
    #     survey_loss = ls.copy()
    #     ################################################################################
    #     # # inspiration from MultiNest: discard points with loss > l_max_prev
    #     # # not useful since acceptance rate too low
    #     # cond = (survey_loss <= l_max_prev)
    #     # survey_params = survey_params[cond]
    #     # survey_loss = survey_loss[cond]
    #     # del cond
    #     # gc.collect()
    #     # if survey_loss.size:
    #     ################################################################################
    #     s_max = np.argmax(survey_loss)
    #     s_min = np.argmin(survey_loss)
    #     l_max = survey_loss.max()
    #     l_min = survey_loss.min()
        
    #     # use 'Mahalonobis distance' a la MultiNest? easier to use current param range as proxy

    #     # adjust survey by deleting outliers and decreasing param range
    #     while survey_loss.size > 10: #0.1*self.N_survey: # make this user-adjustable?  
    #         # iterate to find parameter direction along which current s_max is strongest outlier
    #         positive = []
    #         outliers = []
    #         strengths = []
            
    #         # store current param ranges
    #         param_range = [self.param_maxs[n] - self.param_mins[n] for n in range(self.n_params)]
            
    #         # find all directions along which s_max is an outlier and record strengths
    #         for n in np.arange(self.n_params):
    #             all_but_smax = survey_params[np.arange(survey_params.shape[0]) != s_max,n] # all param values along direction n, except s_max
    #             par_max = 1.0*survey_params[s_max,n]
    #             outlier_strength = np.fabs(par_max - survey_params[s_min,n])/param_range[n] # abs(max_loss_loc - min_loss_loc)/range
    #             if np.all(par_max > all_but_smax):
    #                 outliers.append(n)
    #                 positive.append(1)
    #                 strengths.append(outlier_strength)
    #             if np.all(par_max < all_but_smax):
    #                 outliers.append(n)
    #                 positive.append(0)
    #                 strengths.append(outlier_strength)

    #         # determine strongest outlier direction and adjust its param range
    #         if len(outliers) > 0:
    #             o_max = np.argmax(strengths)
    #             # print('n:',outliers[o_max],'; max-min:',l_max-l_min,
    #             #       '; pos:',positive[o_max],'; value:',survey_params[s_max,outliers[o_max]])
    #             if positive[o_max]:
    #                 self.param_maxs[outliers[o_max]] = survey_params[s_max,outliers[o_max]]
    #             else:
    #                 self.param_mins[outliers[o_max]] = survey_params[s_max,outliers[o_max]]
    #         else:
    #             # worst loss needn't be outlier in any direction. in this case delete the point but don't adjust ranges.
    #             pass
        
    #         # delete s_max
    #         survey_params = np.delete(survey_params,s_max,axis=0)
    #         survey_loss = np.delete(survey_loss,s_max)
    #         # recalculate l_max,l_min,s_max,s_min
    #         s_max = np.argmax(survey_loss)
    #         s_min = np.argmin(survey_loss)
    #         l_max = survey_loss.max()
    #         l_min = survey_loss.min()
            
    #     if self.verbose:
    #         self.print_this('... survey survivors = {0:d}'.format(survey_loss.size),self.logfile)
    #     ################################################################################
    #     ################################################################################
    #     return 
    # ###########################################
