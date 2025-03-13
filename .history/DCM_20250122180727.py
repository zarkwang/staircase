
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import optimize
import pickle
import copy
import os
import multiprocessing as mp
from utils import *


# -----------------------------
#   Discrete Choice Models
# -----------------------------
#  ChoiceModel: a common module
#  mixedDiscrete: latent class model (parameters follow discrete distributions)
#  mixedNormal: parameters follow normal distributions or its related distributions
#  We apply the EM algorithm to fit each model


option_setup = ['x1', 'x2', 'p1', 'p2', 't1', 't2']

'''
Options: 
A: (x1, p1, t1); B: (x2, p2, t2)

Parameters:
    riskCoef: risk aversion / utility curvature
    probW: probability weighting
    temp: temperature / decision noise
    delta: patience / discount factor
'''


class choiceModel:

    def __init__(self,data,choice,x1,x2,
                 p1=1,p2=1,t1=0,t2=0,fixed_args={},func='logitProb'):
        
        self.fixed_args = fixed_args
        self.choice = data[choice].values
        self.func = func
        
        for attr in option_setup:
            new_var = locals()[attr] 
            if isinstance(new_var, str):  
                setattr(self, attr, data[new_var].values)
            elif isinstance(new_var,list) and isinstance(new_var[0],str):
                setattr(self, attr, data[new_var].values)
            else:
                setattr(self, attr, new_var)  


        '''
        x, p, t: str, number, or array
        
        (1) str: specify the correponding column names
        (2) number: specify the corresponding values
        (3) array: for instance, x1 = np.array([10, 20]), p1 = np.array([0.4, 0.6]) imply that 
        "getting 10 with prob. 0.4 and 20 with prob. 0.6"


        choice: str or list
        Specify which columns are the observed choices
        Each choice must be binary (0 or 1). 
        1 = choosing option 1; 0 = choosing choosing option 2.

        fixed_args: dict
        Fixed arguments, such as {'riskCoef':3, 'temp': 2} 
        '''

    def set_init_param(self, param_init=None, param_keys=None, weight=None):

        self.param_init = param_init
        self.param_keys = param_keys
        self.sample_weight = weight

        if 'delta' not in param_keys:
            self.fixed_args = self.fixed_args | {'delta':1}
        
        '''
        param_keys: list 
        Set up keys (names) for the parameters.

        param_init: list 
        Set up intial values for the parameters. Must match to param_keys.

        sample_weight: array
        Sampling weight for each individual
        '''


    def choiceProb(self,params,**kwargs):

        option_attr = {}
        
        for attr in option_setup:
            if attr not in locals():
                option_attr[attr] = getattr(self, attr)

        # Specify the choice probability function
        if self.func == 'logitProb':
            return logitProb(params,option_attr['x1'],option_attr['x2'],
                                    option_attr['p1'],option_attr['p2'],
                                    option_attr['t1'],option_attr['t2'])
        elif self.func == 'logitProb_expanded':
        # considering multiple random draws
        # pecifically for mixedNormal (or continuous distributions)
            return logitProb_expanded(params,option_attr['x1'],option_attr['x2'],
                                    option_attr['p1'],option_attr['p2'],
                                    option_attr['t1'],option_attr['t2'])

    def logLike(self, coefs , w = 1, subsample_idx = None):

        all_args = self.fixed_args | dict(zip(self.param_keys, coefs))

        if subsample_idx is not None:
            y = self.choice[subsample_idx]

            option_attr = {}
            for attr in option_setup:
                var = getattr(self, attr)
                if not np.isscalar(var):
                    option_attr[attr] = var[subsample_idx]
                else:
                    option_attr[attr] = var
            
            p = logitProb(all_args,option_attr['x1'],option_attr['x2'],
                                    option_attr['p1'],option_attr['p2'],
                                    option_attr['t1'],option_attr['t2'])
            
            if self.sample_weight is not None:
                sample_weight = self.sample_weight[subsample_idx]

        else:
            p = self.choiceProb(all_args)
            y = self.choice
            if self.sample_weight is not None:
                sample_weight = self.sample_weight
        
        logLogit = y*np.log(p) + (1-y)*np.log(1-p)

        if len(logLogit.shape) > 1:
            logLogit = logLogit.sum(axis = 1)
        
        if isinstance(w,int) and w == 1:
            w = [w]*len(y)
        
        if self.sample_weight is not None:
            w = w * sample_weight

        return logLogit @ w
        

    def fit_param(self,bounds=None):

        obj = lambda coefs: - self.logLike(coefs)*1000

        x0 = list(self.param_init)

        self.result = optimize.minimize(obj, x0, method='L-BFGS-B', bounds=bounds)

        print(self.result)
    
    def predict_const(self):

        all_args = self.fixed_args | dict(zip(self.param_keys, self.result.x))

        self.pred = self.choiceProb(all_args) > 0.5

        accuracy = np.sum((self.pred - self.choice) == 0) / self.choice.size

        return {'accuracy': accuracy}
    

class mixedDiscrete(choiceModel):

    def set_init_mixed(self, latent_class: dict, latent_share: list,
                       weight = None):
        self.latent_class = latent_class
        self.latent_share = latent_share

        self.param_keys = list(latent_class.keys())
        self.n_class = len(list(latent_class.values())[0])
        self.obj_values = [0]*self.n_class

        if 'delta' not in self.param_keys:
            self.fixed_args = self.fixed_args | {'delta':1}
        
        if weight is not None:
            self.sample_weight = weight


    def gen_init_params(self, n_class, sample_ratio = 0.3, bounds = None, show = False):

        latent_class = {key: [] for key in self.param_keys}

        for i in range(n_class):
            # draw a random sample
            indiv_idx = np.arange(len(self.choice))
            subsample_idx = np.random.choice(indiv_idx, size=round(len(self.choice)*sample_ratio),replace=True)

            # fit a model using the random sample, to generate parameter values
            obj = lambda coefs: - self.logLike(coefs,subsample_idx=subsample_idx)
            x0 = list(self.param_init)

            result = optimize.minimize(obj, x0, method='L-BFGS-B', bounds=bounds)

            for k in range(len(self.param_keys)):
                latent_class[self.param_keys[k]] += [result.x[k]]
        
        self.set_init_mixed(latent_class = latent_class, 
                            latent_share = [1/n_class]*n_class)
        
        if show:
            print('Initial setting:', self.latent_class)


    def likeIndiv(self,latent_class: dict):

        mat_shape = np.append(np.array(self.choice.shape),0)

        prob = np.empty(tuple(mat_shape))

        for i in range(self.n_class):

            param_each_class = [value[i] for value in latent_class.values()]
            new_args = dict(zip(self.param_keys,param_each_class))

            all_args = self.fixed_args | new_args
        
            p = self.choiceProb(all_args)
            y = self.choice
            logLogit = y*p + (1-y)*(1-p)
            prob = np.concatenate((prob, logLogit[:, :, np.newaxis]), axis=2) 

        return prob

 
    def postProbLatent(self):

        # m × n matrix
        # m: number of latent classes
        # n: number of individuals
        choiceProb = self.likeIndiv(self.latent_class)

        new_share = np.array(self.latent_share)
        
        jointProb = choiceProb.prod(axis=1) * new_share[np.newaxis,:]

        # Conditional on person i choosing j, the probability that the person is in latent class b
        post_prob = jointProb / jointProb.sum(axis=1, keepdims=True)

        return post_prob
    

    def updateShare(self):

        post_prob = self.postProbLatent()
        new_share = post_prob.sum(axis=0) / post_prob.sum()

        self.post_prob = post_prob
        self.latent_share = new_share

        return new_share
    

    def fitEachClass(self,bounds=None):

        for c in range(self.n_class):

            w = self.post_prob[:,c]
            param_init = [value[c] for value in self.latent_class.values()]

            obj = lambda coefs: - self.logLike(coefs,w)

            result = optimize.minimize(obj, param_init, method='L-BFGS-B', bounds=bounds)

            for key,value in zip(self.latent_class.keys(),result.x):
                self.latent_class[key][c] = value
            
            self.obj_values[c] = result.fun

            if np.isnan(result.fun):
                self.obj_values[c] = obj(result.x)

        return self.obj_values
    
    
    def runEM(self,max_iter=300,tol=1e-3,bounds=None,show=False):

        # expectation-maximization (EM) algorithm
        self.updateShare()
        old_obj = self.obj_values.copy()
        new_obj = self.fitEachClass(bounds)

        new_share = np.round(self.latent_share, 3)
        new_param = dict(zip(self.param_keys, 
                                np.round(list(self.latent_class.values()), 3)))
        i = 1

        if show:
            print('Iteration', i)
            print('Class share', new_share)
            print('Class parameter', new_param)
            print('objective', np.round(new_obj,3))

        while i < max_iter and abs(sum(new_obj) - sum(old_obj)) > tol:
            self.updateShare()
            old_obj = self.obj_values.copy()
            new_obj = self.fitEachClass(bounds)

            new_share = np.round(self.latent_share, 3)
            new_param = dict(zip(self.param_keys, 
                                 np.round(list(self.latent_class.values()), 3)))
            
            i += 1
            if show:
                print('Iteration', i)
                print('Class share', new_share)
                print('Class parameter', new_param)
                print('objective', np.round(new_obj,3))
    

    def eval(self):

        choiceProb = self.likeIndiv(self.latent_class).prod(axis=1)

        latent_share = np.array(self.latent_share)

        eval_func = - choiceProb.sum(axis=0) @ np.log(latent_share) + sum(self.obj_values)

        return eval_func
    
    def predict(self):

        if hasattr(self, 'latent_class') :
            new_like = self.likeIndiv(self.latent_class) * self.post_prob[:,np.newaxis]
            y = self.choice[:,np.newaxis]
            loglike = (np.log(new_like) * y  + np.log(1 - new_like) * (1-y))

            self.pred = new_like.sum(axis=2) > 0.5

            accuracy = np.sum((self.pred - self.choice) == 0) / self.choice.size
             

            pred_result = {'accuracy': accuracy,'cross-entropy': - loglike}
        else:
            pred_result = self.predict_const()

        return pred_result


# Model fitting results depend on initial points
# We randomly generate n_init_point of them and select the best-performing result
# Find the best model given constant args
def run_model_once(args):
    """
    Run the model once and return the evaluation score and the model.
    Args is a tuple: (model, bounds, fixed_args, n_class)
    """
    model, bounds, fixed_args, n_class, max_iter = args
    model.gen_init_params(n_class=n_class, sample_ratio=0.3, bounds=bounds)
    model.fixed_args.update(fixed_args)
    model.runEM(bounds=bounds,max_iter=max_iter)
    new_eval = model.eval()
    return new_eval, model


def get_best_result(model, bounds, n_class, n_init_point, 
                    name_prefix, name_suffix=None, update_args = {}, n_jobs=4, max_iter =300):
    """
    Run the model multiple times in parallel using multiprocessing and select the best result.
    """
    # Prepare arguments for each process
    task_args = [(copy.deepcopy(model), bounds, update_args, n_class, max_iter) for _ in range(n_init_point)]

    # Use multiprocessing Pool
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(run_model_once, task_args)

    # Find the best result
    min_eval, best_model = min(results, key=lambda x: x[0])

    # Save the best model
    file_name = f"{name_prefix}_class_{n_class}"
    if name_suffix:
        file_name += f"_{name_suffix}.pkl"
    else:
        file_name += f".pkl"

    with open(file_name, 'wb') as f:
        pickle.dump(best_model, f)

    return min_eval


def bisection_search(model, bounds, n_class, n_init_point, 
                     name_prefix, arg_name, arg_range, tol = 1e-3, max_iter =300, stop_search = 10):
    '''
    Search for the value of a constant arg, to minimize the model evaluation metric
    '''
    lower = arg_range[0] 
    upper = arg_range[1]
    i = 0

    print(f'Searching optimal value for {arg_name}')

    while i<stop_search and (upper - lower > tol):
        m1 = lower + (upper - lower) / 3
        m2 = upper - (upper - lower) / 3

        args1 = {arg_name: m1}
        args2 = {arg_name: m2}

        eval_func_1 = get_best_result(model,bounds,n_class,n_init_point,name_prefix,
                                      name_suffix=1,update_args=args1)
        eval_func_2 = get_best_result(model,bounds,n_class,n_init_point,name_prefix,
                                      name_suffix=2,update_args=args2)

        if eval_func_1 < eval_func_2:
            upper = m2
        else:
            lower = m1
        
        print(f'Step {i+1}: {(lower + upper) / 2}')
        print(f'Evaluation metrics: {eval_func_1}, {eval_func_2}')

        i+=1

    new_arg = {arg_name: (lower + upper) / 2}

    min_func = get_best_result(model,bounds,n_class,n_init_point,name_prefix,
                               update_args=new_arg,max_iter=max_iter)

    os.remove(f'{name_prefix}_class_{n_class}_1.pkl')
    os.remove(f'{name_prefix}_class_{n_class}_2.pkl')

    return new_arg, min_func



class mixedNormal(choiceModel):
    def set_init_mixed(self, meta_param: dict, param_keys: list, dist_type: list, 
                            n_draws, n_batch = 1, weight = None,
                            constant_param = {}):
        self.meta_param = meta_param
        self.constant_param = constant_param
        self.dist_type = dist_type
        self.param_keys = param_keys
        self.n_draws = n_draws
        self.n_batch = n_batch
        self.sample_weight = weight
        self.func = 'logitProb_expanded'

        if 'delta' not in self.param_keys:
            self.fixed_args = self.fixed_args | {'delta':1}
        
        if 'probW' not in self.param_keys:
            self.fixed_args = self.fixed_args | {'probW':1}
        
        self.n_random = n_draws * n_batch
        self.random_params = {key: None for key in param_keys}
        
        '''
        latent_param: 
        Specify the mean and covariance, e.g. {'mean':[1,2],'variance':[[1,0],[0,2]]}

        param_keys:
        Specify the parameter names, e.g. ['riskCoef','delta','temp'] 
        Must be matched to latent_param.
        
        dist_type: 
        Specify the types for distribution, e.g. ['normal','johnson_sb']

        Note there could be multiple parameters and we estimate their joint distribution. 
        '''

    @staticmethod
    def dist_transform(x,dist_type='normal'):
        if dist_type == 'normal':
            return x
        elif dist_type == 'lognormal':
            return np.exp(x)
        elif dist_type == 'johnson_sb':
            return 1/(1+np.exp(-x))
    

    def gen_random_params(self,n_draws,clear=False,optim=False):

        if optim == False:
            _mean = np.array(self.meta_param['mean'])
            _covar = np.array(self.meta_param['variance'])
        else:
            _mean = np.array(self.optim_metaparam['mean'])
            _covar = np.array(self.optim_metaparam['variance'])

        n_params = len(_mean)
        n_indiv = len(self.choice)
        n_total_draws = n_indiv * n_draws

        transformed_params = {}

        normal_samples = np.random.normal(loc=0, scale=1, size=(n_total_draws, n_params), )

        # Compute the Cholesky factor
        if len(_covar) == 1:
            L = np.sqrt(_covar)
        else:
            L = np.linalg.cholesky(_covar)

        # Compute the random parameter values
        if len(_covar) == 1:
            random_params = np.tile(_mean, (n_total_draws, 1)) + normal_samples * L
        else:
            random_params = np.tile(_mean, (n_total_draws, 1)) + normal_samples @ L

        # Record all generated random parameter values in self.random_params
        # Transform normal to other distributions

        if clear == True:
            self.random_params = {key: None for key in self.param_keys}

        for i in range(len(self.dist_type)):
            key = self.param_keys[i]
            new_params = random_params[:,i]

            if self.random_params[key] is None:
                self.random_params[key] = new_params.reshape(n_indiv, n_draws)
            else:
                self.random_params[key] = np.hstack((self.random_params[key], 
                                                     new_params.reshape(n_indiv, n_draws)))

            transformed = self.dist_transform(new_params,self.dist_type[i])
            transformed_params[self.param_keys[i]] = transformed.reshape(n_indiv, n_draws)
        
        return transformed_params
    

    def genAllParams(self,constant_params=None):

        all_params = self.gen_random_params(self.n_draws).copy()

        n_indiv = len(self.choice)
        n_total_draws = n_indiv * self.n_draws

        for key,value in self.fixed_args.items():
            all_params[key] = np.repeat(value,n_total_draws).reshape(n_indiv,self.n_draws)

        if constant_params is not None:
            const_keys = list(self.constant_param.keys())
            for i in range(len(const_keys)):
                all_params[const_keys[i]] = np.repeat(constant_params[i],
                                                      n_total_draws).reshape(n_indiv,self.n_draws)
        
        return all_params

    
    def likeIndivBatch(self,constant_params=None):
        
        all_params = self.genAllParams(constant_params)

        p = self.choiceProb(all_params)
        y = np.tile(self.choice,(self.n_draws,1,1)).T
        likeFunc = y*p + (1-y)*(1-p)
        
        return likeFunc.prod(axis = 0)
    

    def likeIndiv(self,constant_params=None):

        i = 1
        like_indiv = self.likeIndivBatch(constant_params)

        while i < self.n_batch:
            new_like = self.likeIndivBatch(constant_params)
            like_indiv = np.hstack((like_indiv, new_like))
            i += 1
        
        return like_indiv
    
    def objEM(self,const):

        obj_func = - np.sum(self.w * np.log(self.likeIndiv(const))) / self.n_random

        return obj_func
    
    
    def updateMetaParam(self):

        const = list(self.constant_param.values())

        if len(const) == 0:
            like = self.likeIndiv()
        else:
            like = self.likeIndiv(const)
        
        if self.sample_weight is not None:
            self.w = self.n_random * like / np.sum(like, axis=1, keepdims=True) * self.sample_weight[:,np.newaxis]
        else:
            self.w = self.n_random * like / np.sum(like, axis=1, keepdims=True)
        # shape of w: (n_indiv, n_random)

        for i in range(len(self.param_keys)):
            self.meta_param['mean'][i] = np.mean(self.random_params[self.param_keys[i]] * self.w)

        params_deviation = np.stack(list(self.random_params.values()), axis=-1) - self.meta_param['mean']

        # params_deviation takes this form: (i,j,k)
        # Use np.einsum to perform outer products
        # This is identical to np.outer(params_deviation[i,j],params_deviation[i,j]) for all (i,j)
        out = np.einsum('ijk,ijl->ijkl', params_deviation, params_deviation)

        w_expanded = np.expand_dims(np.expand_dims(self.w, axis=-1), axis=-1)

        self.meta_param['variance'] = np.mean(out * w_expanded, axis=(0, 1))

        self.getGradient(params_deviation,out)

        self.obj_func = self.objEM(const)

        return self.meta_param
    

    def getGradient(self,params_deviation,out):

        n_key = len(self.param_keys)
        n_indiv = len(self.choice)

        # Compute the gradients
        inv_covar = np.linalg.inv(self.meta_param['variance'])

        mean_scores = np.empty((n_indiv,n_key))
        covar_scores = np.empty((n_indiv,n_key,n_key))

        # Expand w to the required shapes
        w_expanded = self.w[:, :, np.newaxis]  # Shape becomes (n_indiv, n_random, 1)
        w_expanded_2 = w_expanded[:, :, np.newaxis]  # Shape becomes (n_indiv, n_random, 1, 1)

        mean_scores = np.sum(
                -w_expanded * (params_deviation @ inv_covar), axis=1
                ) / self.n_random  # Shape: (n_indiv, n_param)
        

        inv_covar_out_inv_covar = inv_covar @ out @ inv_covar.T - inv_covar

        covar_scores = np.sum(
                w_expanded_2 * 0.5 * inv_covar_out_inv_covar,
                axis=1,
                ) / self.n_random  # Shape: (n_indiv, n_param)

        # for i in range(n_indiv):
        #     w_i = self.w[i,:][:, np.newaxis]
        #     w_ii = w_i[:, np.newaxis]
        #     mean_scores[i] = np.sum(- w_i * (params_deviation[i,:,:] @ inv_covar)) / self.n_draws
        #     covar_scores[i] = np.sum(w_ii * 0.5 * (inv_covar @ out @ inv_covar.T - inv_covar)[i,:]) / self.n_draws

        mean_grad = mean_scores.sum(axis = 0)
        covar_grad = covar_scores.sum(axis = 0)

        self.gradients = (mean_grad, covar_grad)

        return self.gradients
    

    def updateConstantParam(self,bounds):
        
        init = list(self.constant_param.values())

        result = optimize.minimize(self.objEM, init, method='L-BFGS-B', bounds=bounds)

        self.constant_param = dict(zip(self.constant_param.keys(), result.x))

        return result.fun
    

    def runEM(self,min_iter=1,max_iter=500,tol=1,excess_step=30,bounds=None):

        # expectation-maximization (EM) algorithm
        i = 1
        old_const_obj = 0

        self.updateMetaParam()
        new_obj = self.obj_func

        print('Iteration', i)
        print(self.meta_param)
        print('gradient:',self.gradients)
        print('objective:',new_obj)

        self.optim_i = i
        self.optim_func = new_obj
        self.optim_w = copy.deepcopy(self.w)
        self.optim_metaparam = copy.deepcopy(self.meta_param)

        const = list(self.constant_param.values())
        

        if len(const) > 0:
            new_const_obj = self.updateConstantParam(bounds)
            print('constants:',self.constant_param)
            print('const_obj:',new_const_obj)
        else:
            new_const_obj = 100
        
        self.random_params = {key: None for key in self.param_keys}

        while i < min_iter or \
            (i < max_iter and i < (self.optim_i + excess_step) and 
            new_obj < (self.optim_func + tol) or new_obj == np.inf) or \
            (abs(new_const_obj - old_const_obj) < tol):
             
            i += 1

            self.updateMetaParam()
            new_obj = self.obj_func

            print('Iteration', i)
            print(self.meta_param)
            print('gradient:',self.gradients)
            print('objective:',new_obj)
           
            if new_obj < self.optim_func:
                self.optim_i = i
                self.optim_func = new_obj
                self.optim_w = copy.deepcopy(self.w)
                self.optim_metaparam = copy.deepcopy(self.meta_param)
                

            if len(const) > 0:
                old_const_obj = new_const_obj
                new_const_obj = self.updateConstantParam(bounds)
                print('constants:',self.constant_param)
                print('const_obj:',new_const_obj)
            else:
                new_const_obj = old_const_obj + 100
            
            self.random_params = {key: None for key in self.param_keys}

    
    def simuIndivParam(self,optim=False):
        
        self.gen_random_params(self.n_random, clear=True, optim=optim)

        indiv_params = {}

        if optim:
            _w = self.optim_w
        else:
            _w = self.w

        if self.sample_weight is not None:
            w = _w / self.sample_weight[:,np.newaxis]
        else:
            w = _w

        for i in range(len(self.param_keys)):
            key = self.param_keys[i]
            new_simu_params = self.dist_transform(self.random_params[key],self.dist_type[i])
            indiv_params[key] = (new_simu_params * w / self.n_random).sum(axis=1)
        
        self.simu_indiv_params = indiv_params

        return indiv_params
    





        


        



    


    
    


        














    