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

        # uncomment block below to get inv cov mat focus
        # invcov_mat = params.get('invcov_mat',None)
        # self.invcov_mat = np.eye(self.Y_full.shape[1]) if invcov_mat is None else invcov_mat
        # if self.invcov_mat.shape != (self.Y_full.shape[1],self.Y_full.shape[1]):
        #     raise ValueError('Incompatible invcov_mat in Chi2.')
        # # this is full inv cov mat, to be passed only once by user in initial parameter dictionary

        self.subset = params.get('subset',np.s_[:]) # slice or array of indices, used to extract (shuffled) training or validation elems.
        self.Y_sub = self.Y_full[:,self.subset].copy()
        self.n_sub = self.Y_sub.shape[1] # n_sub = n_samp for this data set

        self.cov_mat_sub = self.cov_mat[:,self.subset][self.subset,:].copy()
        self.L_sub = linalg.cholesky(self.cov_mat_sub,lower=True) # so C = L L^T
        # uncomment below to get inv cov mat focus
        # self.invcov_mat_sub = self.invcov_mat[:,self.subset][self.subset,:].copy()
        
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
        # uncomment below to get inv cov mat focus
        # self.Loss_vec = np.dot(self.invcov_mat_sub,resid_sub.T)*self.batch_mask # (n_samp,1) with non-zero only in batch
        Loss = np.dot(self.resid_b,self.Loss_vec) # (1,1)
        return Loss[0,0] # scalar

    def backward(self):
        dLdm = self.Loss_vec.T # (1,n_samp)
        z = self.rv(linalg.cho_solve((self.L_sub,True),self.resid_b[0],check_finite=False))
        # solves (L L^T) z^T = resid or z = (C^-1 resid)^T, shape (1,n_samp)
        dLdm += z
        # uncomment below to get inv cov mat focus
        # dLdm += np.dot(self.resid_b,self.invcov_mat_sub) # (1,n_samp) x (n_samp,n_samp) = (1,n_samp)
        return dLdm

    # uncomment two blocks below to get inv cov mat focus
    # def forward(self,Ypred_sub):
    #     resid_sub = Ypred_sub - self.Y_sub # (1,n_samp)
    #     self.resid_b = resid_sub*self.batch_mask # (1,n_samp) with non-zero only in batch
    #     self.Loss_vec = np.dot(self.invcov_mat_sub,resid_sub.T)*self.batch_mask # (n_samp,1) with non-zero only in batch
    #     Loss = np.dot(self.resid_b,self.Loss_vec) # (1,1)
    #     return Loss[0,0] # scalar

    # def backward(self):
    #     dLdm = self.Loss_vec.T # (1,n_samp)
    #     dLdm += np.dot(self.resid_b,self.invcov_mat_sub) # (1,n_samp) x (n_samp,n_samp) = (1,n_samp)
    #     # dLdm = 2*self.resid_b # (1,n_samp) 
    #     return dLdm


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
                chk_half = (self.val_loss[t] > self.val_loss[t-check_after//2])
                chk = (self.val_loss[t-check_after//2] > self.val_loss[t-check_after])
                if chk_half & chk:
                    if self.verbose:
                        self.print_this('',self.logfile)
                    break
                
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
            
            ## Adam params will eventually be hidden ##
            -- 'adam': bool, whether or not to implement adam support. ** Strongly recommend True. **
            -- 'B1_adam': float, adam parameter for momentum. Default 0.9.
            -- 'B2_adam': float, adam parameter for adadelta. Default 0.999.
            -- 'eps_adam': float, adam parameter for zero-division safety. Default 1e-8.
            ##

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
            -- 'max_epoch': int, number of epochs (equiv: loss calls or model evaluations) per mini-batch. Default 30.
            -- 'mb_count': int, number of mini-batches. Default 3.
            -- 'lrate': float, SGD learning rate. Default 0.1. # CAN THIS BE NON-DIMENSIONALISED?
            -- 'loss_params': dictionary with common keys to be used by all Walker instances. 
                              E.g. for Chi2 this would be 'cov_mat' containing covariance matrix of *full* data set.
            -- 'check_after': int, number of epochs after which to check for increase in validation loss. Default 10.

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
        self.Model = data_pack.get('model',None)
        self.n_params = data_pack.get('n_params',None)
        self.param_mins = data_pack.get('param_mins',None)
        self.param_maxs = data_pack.get('param_maxs',None)

        self.N_survey = int(self.survey_frac*self.N_evals_max)
        self.N_evals_max_walk = self.N_evals_max - self.N_survey
        
        # adam setup
        self.adam = data_pack.get('adam',True)
        self.B1_adam = data_pack.get('B1_adam',0.9)
        self.B2_adam = data_pack.get('B2_adam',0.999)
        self.eps_adam = data_pack.get('eps_adam',1e-8)

        self.X = data_pack.get('X',None)
        self.Y = data_pack.get('Y',None)
        
        self.check_init()
        
        self.model_inst = self.Model(n_params=self.n_params,adam=self.adam,
                                     B1_adam=self.B1_adam,B2_adam=self.B2_adam,eps_adam=self.eps_adam)
        
        self.val_frac = data_pack.get('val_frac',0.2) # fraction of input data to use for validation
        
        self.loss_module = data_pack.get('loss',Chi2) 
        self.seed = data_pack.get('seed',None)
        self.walks_exist = data_pack.get('walks_exist',False)
        self.verbose = data_pack.get('verbose',True)
        self.logfile = data_pack.get('logfile',None)

        if self.verbose:
            self.print_this('Mountaineer to explore loss land-scape!',self.logfile)
            
        self.loss_params = copy.deepcopy(data_pack.get('loss_params',{}))

        # loss difference threshold for survey. what is a principled way of setting this?
        # NOT ROBUST TO VERY BROAD INITIAL RANGE
        self.Delta_loss_threshold = 20.0*(self.X.shape[1] - self.n_params)
        
        self.distributed = 0 # will be set to 1 (-1) upon (un)successful call to self.distribute().
        
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

        if self.N_survey > 0:
            if self.verbose:
                self.print_this('Surveying using {0:d} locations...'.format(self.N_survey),self.logfile)
        else:
            if self.verbose:
                self.print_this('Skipping survey...',self.logfile)
            return
            
        model_survey = copy.deepcopy(self.model_inst) # AVOID COPYING IF POSSIBLE
        survey_params = self.gen_latin_hypercube(Nsamp=self.N_survey,dim=self.n_params,
                                                       param_mins=self.param_mins,param_maxs=self.param_maxs)
        # survey_params has shape (N_survey,n_params)

        lp = copy.deepcopy(self.loss_params)
        lp['Y_full'] = self.Y.copy() # since Walker is not invoked
        loss = self.loss_module(params=lp)
        loss_survey = np.zeros(self.N_survey)
        for s in range(self.N_survey):
            model_survey.params = survey_params[s:s+1,:].T # shape (n_params,1)
            # calculate total loss at survey parameters
            Ypred = model_survey.forward(self.X)
            loss_survey[s] = loss.forward(Ypred)
            # write this somewhere so as to not lose evaluations !
            if self.verbose:
                self.status_bar(s,self.N_survey)

        self.param_maxs_old = copy.deepcopy(self.param_maxs)
        self.param_mins_old = copy.deepcopy(self.param_mins)
        
        s_max = np.argmax(loss_survey)
        l_max = loss_survey.max()
        l_min = loss_survey.min()
        
        # NOT ROBUST TO VERY BROAD INITIAL RANGE
        while (loss_survey.size > 1) & ((l_max - l_min) > self.Delta_loss_threshold):
            # find one parameter direction along which s_max is furthest out
            positive = 1
            for n in range(self.n_params):
                all_but_smax = survey_params[np.arange(survey_params.shape[0]) != s_max,n]
                if np.all(survey_params[s_max,n] > all_but_smax):
                    break
                if np.all(survey_params[s_max,n] < all_but_smax):
                    positive *= 0
                    break
            # modify param_mins or param_maxs for furthest (last value of n)
            if positive:
                self.param_maxs[n] = 0.5*(survey_params[s_max,n] + all_but_smax.max()) # between largest and 2nd largest
                # self.param_maxs[n] = survey_params[s_max,n] - 0.05*np.abs(survey_params[s_max,n]) # just below largest
            else:
                self.param_mins[n] = 0.5*(survey_params[s_max,n] + all_but_smax.min()) # between smallest and 2nd smallest
                # self.param_mins[n] = survey_params[s_max,n] + 0.05*np.abs(survey_params[s_max,n]) # just above smallest
                
            # delete s_max
            survey_params = np.delete(survey_params,s_max,axis=0)
            loss_survey = np.delete(loss_survey,s_max)
            # recalculate l_max,l_min,s_max
            s_max = np.argmax(loss_survey)
            l_max = loss_survey.max()
            l_min = loss_survey.min()

        if self.verbose:
            prnt_str = '... old param_maxs  = ['+','.join(['{0:.2e}'.format(p) for p in self.param_maxs_old]) +']\n'
            prnt_str += '... ... modified to = ['+','.join(['{0:.2e}'.format(p) for p in self.param_maxs]) +']\n'
            prnt_str += '... old param_mins  = ['+','.join(['{0:.2e}'.format(p) for p in self.param_mins_old]) +']\n'
            prnt_str += '... ... modified to = ['+','.join(['{0:.2e}'.format(p) for p in self.param_mins]) +']'
            self.print_this(prnt_str,self.logfile)
            
        return
    ###########################################

    ###########################################
    def distribute(self):
        """ Distribute available resources among walkers and set training parameters. 
            Returns +1 for successful distribution and -1 if not enough walkers can be initialised.
        """
        if self.walks_exist:
            raise Exception('Walks exist! No distribution allowed.')
        
        self.N_walker = 2*self.n_params # ?? 10*(n_params/3)

        if self.N_evals_max_walk <= self.N_walker:        
            if self.verbose:
                self.print_this('{0:d} evals is not enough for exploration.'.format(self.N_evals_max_walk),self.logfile)
            self.distributed = -1
            return             
        
        if self.verbose:
            self.print_this('Distributing available resources ({0:d} evals)...'.format(self.N_evals_max_walk),self.logfile)
            
        if self.verbose:
            self.print_this('... initialising {0:d} walkers'.format(self.N_walker),self.logfile)
        
        dp_walk = {'X':self.X,'Y':self.Y,'val_frac':self.val_frac,
                   'model':self.model_inst,'loss':self.loss_module,'walks_exist':self.walks_exist,
                   'seed':self.seed,'verbose':False,'logfile':self.logfile}

        self.walkers = []
        self.pins = self.gen_latin_hypercube(Nsamp=self.N_walker,dim=self.n_params,
                                        param_mins=self.param_mins,param_maxs=self.param_maxs)
        for w in range(self.N_walker):
            dp_walk['file_stem'] = self.file_stem + '_w' + str(w+1)
            dp_walk['params_init'] = self.pins[w:w+1,:].T
            self.walkers.append(Walker(data_pack=dp_walk))
            
        if self.verbose:
            self.print_this('... setting up training parameters',self.logfile)
            
        ####################
        # these two maybe can be put in __init__
        self.mb_count = int(np.sqrt(self.X.shape[1])) # 3
        self.lrate = 0.1 # check sensitivity to this
        ####################
        self.max_epoch = self.N_evals_max_walk // (self.N_walker*self.mb_count) # 30
        self.check_after = self.max_epoch // 3
        if self.check_after < 2:
            self.check_after = self.max_epoch + 1
            
        if self.verbose:
            self.print_this('... ...   max_epoch = {0:d}'.format(self.max_epoch),self.logfile)
            self.print_this('... ...    mb_count = {0:d}'.format(self.mb_count),self.logfile)
            self.print_this('... ... check_after = {0:d}'.format(self.check_after),self.logfile)
            self.print_this('... ...       lrate = {0:.3f}'.format(self.lrate),self.logfile)

        self.params_train = {'max_epoch':self.max_epoch,
                             'mb_count':self.mb_count,
                             'lrate':self.lrate,
                             'loss_params':self.loss_params,
                             'check_after':self.check_after}
        self.distributed = 1
        return 
    ###########################################

    ###########################################
    def climb(self):
        """ Train all walkers and store outputs. Should be called after distribute. """
        if self.walks_exist:
            raise Exception('Walks exist! No climbing allowed.')
        if self.distributed == 0:
            raise Exception('Distribution not done! No climbing allowed.')
        if self.distributed == -1:
            raise Exception('Distribution not successful! No climbing allowed.')
        
        if self.verbose:
            self.print_this('Climbing...',self.logfile)

        for w in range(self.N_walker):
            self.walkers[w].train(params=self.params_train)
            if self.verbose:
                self.status_bar(w,self.N_walker)
        
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
            
        steps = 0
        
        for w in range(self.N_walker):
            if walks[w].size > 0:
                steps_this = walks[w].shape[1]
                steps += steps_this
                if not self.walks_exist:
                    for s in range(steps_this):
                        self.write_to_file(self.walks_file,list(walks[w][:,s]))
            
        if self.verbose:
            self.print_this('... total steps taken = {0:d}'.format(steps),self.logfile)
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
        if self.distribute == -1:
            raise Exception("Exploration seems to have failed. Can't visualize non-existent walks!")
        
        if self.verbose:
            self.print_this('Visualizing...',self.logfile)
            self.print_this('... setting up visualization colors',self.logfile)
        self.cols = iter(plt.cm.Spectral_r(np.linspace(0,1,self.N_walker)))

        cols = copy.deepcopy(self.cols)
        plt.figure(figsize=(5,5))
        plt.xlabel('x')
        plt.ylabel('y')
        errors = np.sqrt(np.diag(self.loss_params['cov_mat']))
        # uncomment below to get inv cov mat focus
        # errors = 1/np.sqrt(np.diag(self.loss_params['invcov_mat']))
        imin = self.Y[0].argmin()
        imax = self.Y[0].argmax()
        plt.ylim(self.Y[0][imin]-1.5*errors[imin],self.Y[0][imax]+1.5*errors[imax])
        plt.errorbar(self.X[0],self.Y[0],yerr=errors,ls='none',c='k',capsize=5,marker='o')
        for w in range(self.N_walker):
            col = next(cols)
            plt.plot(self.X[0],self.walkers[w].predict(self.X)[0],'-',color=col,lw=1)
        plt.show()

        cols = copy.deepcopy(self.cols)
        fig,ax1 = plt.subplots(figsize=(5,5))
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
                plt.figure(figsize=(5,5))
                plt.xlabel('$a_{{{0:d}}}$'.format(pi))
                plt.ylabel('$a_{{{0:d}}}$'.format(pj))
                xmin,xmax = self.param_mins[pi],self.param_maxs[pi]
                xmin -= 0.1*np.fabs(xmin)
                xmax += 0.1*np.fabs(xmax)
                plt.xlim(xmin,xmax)
                ymin,ymax = self.param_mins[pj],self.param_maxs[pj]
                ymin -= 0.1*np.fabs(ymin)
                ymax += 0.1*np.fabs(ymax)
                plt.ylim(ymin,ymax)
                for w in range(self.N_walker):
                    col = next(cols)
                    plt.plot(walks[w][pi+1],walks[w][pj+1],ls='-',color=col,marker='o',markersize=3,lw=1)
                plt.show()

        return
    ###########################################
