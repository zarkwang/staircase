
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import optimize

# -----------------------------
#    Some useful functions
# -----------------------------

# CRRA utility function
# Note riskCoef ≠ 1. I suggest using 2 as the initial value.
def uCRRA(x,coef):
    return np.array(x)**coef

# exponential discount function
def discountExp(t,coef):
    return coef**np.array(t)

# choice probability under the logit model
def logitProb2(params,x1,x2,p1=1,p2=1,t1=0,t2=0):
        
        '''
        Each choice involves two options
            option 1: (x1,t1,p1)
            option 2: (x2,t2,p2)
        x is the amounts, t is the reward delivery times, p is the probabilities

        u_diff = u(x1,t1,p1) - u(x2,t2,p2)

        choice probability for option 1: 
            P(A) = sigmoid( u_diff / temp )
        where temp is the temperature parameter
    '''

        u_1 = uCRRA(x1,params['riskCoef']) * p1 * discountExp(t1,params['delta'])
        u_2 = uCRRA(x2,params['riskCoef']) * p2 * discountExp(t2,params['delta'])

        if len(u_1.shape) > 2:
            u_1 = u_1.sum(axis=2)
        
        if len(u_2.shape) > 2:
            u_2 = u_2.sum(axis=2)

        u_diff = u_1 - u_2

        prob = 1/(1+ np.exp(-u_diff / params['temp']))

        return prob


# Proportion test
def run_prop_test(node_count):
    # Number of observations in each node
    # hi: higher level for option B (the variable option)
    # lo: lower level for option B
    n_lo = node_count[0] + node_count[1]
    n_hi = node_count[2] + node_count[3]
    n = n_lo + n_hi

    # Number of observations choosing option A (the fixed option)
    q_lo = node_count[1]
    q_hi = node_count[3]
    q = n_hi

    # p(g): proportion of choosing option A in each group, where g is selected from {lo,hi}
    # p(0) denotes the parent node
    # Null Hypothesis: p(lo) >= p(0) >= p(hi)
    # Fisher's exact test
    matrix_lo = [[q_lo, q], [n_lo - q_lo, n - q]]
    matrix_hi = [[q, q_hi], [n - q, n_hi - q_hi]]
    matrix_lo_hi = [[q_lo, q_hi], [n_lo - q_lo, n_hi - q_hi]]

    t1 = st.fisher_exact(matrix_lo, alternative='less')
    t2 = st.fisher_exact(matrix_hi, alternative='less')
    t3 = st.fisher_exact(matrix_lo_hi, alternative='less')

    # Extract p-values
    p_value_1 = t1.pvalue
    p_value_2 = t2.pvalue
    p_value_3 = t3.pvalue

    # Hochberg step-up correction
    sorted_p_values = sorted([p_value_1, p_value_2])
    p_value_up = min(min(np.array(sorted_p_values) * [2, 1]), 1)

    # Holm step-down method
    # p_value_down = min(max(np.array(sorted_p_values) * [2, 1]), 1)

    # Create a new row
    new_row = {
        "n": n,
        "prop_lo": q_lo / n_lo if n_lo > 0 else np.nan,
        "prop_0": q / n if n > 0 else np.nan,
        "prop_hi": q_hi / n_hi if n_hi > 0 else np.nan,
        "p_lo_0": p_value_1,
        "p_0_hi": p_value_2,
        "p_lo_hi": p_value_3,
        "p_up": p_value_up,
        # "p_down": p_value_down,
    }

    return new_row

# -----------------------------
#   Discrete Choice Models
# -----------------------------

option_setup = ['x1', 'x2', 'p1', 'p2', 't1', 't2']

class choiceModel:

    def __init__(self,data,choice,x1,x2,
                 p1=1,p2=1,t1=0,t2=0,fixed_args={'delta':1},func='logitProb2'):
        
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
        x, p, t: 
        (1) Each element be a str: specify the names of correponding columns
        (2) Each element be a number: specific their values
        (3) Each element be a turple: x1 = (10, 20), p1 = (0.5, 0.5) means 
        that "getting 10 or 20 with equal probability"


        choice: str or list
        Specify the names of columns which are the observed choices
        Each choice must be binary (0 or 1). 1 = choosing option 1; 0 = choosing choosing option 2.

        fixed_args: dict
        Fixed arguments, such as {'riskCoef':3, 'temp': 2} 
        '''

    def set_init_param(self, param_init=None, param_keys=None):

        param_setup = {
                "param_init": param_init,
                "param_keys": param_keys
            }
        
        for key, value in param_setup.items():
            if value is not None:
                setattr(self, key, np.array(value))

        '''
        param_keys: list 
        Set up the keys for the parameters.

        param_init: list 
        Set up the intial values for the parameters. Must be matched to param_keys.
        '''


    def choiceProb(self,params,**kwargs):

        option_attr = {}
        
        for attr in option_setup:
            if attr not in locals():
                option_attr[attr] = getattr(self, attr)

        # Specify the choice probability function
        if self.func == 'logitProb2':
            return logitProb2(params,option_attr['x1'],option_attr['x2'],
                                    option_attr['p1'],option_attr['p2'],
                                    option_attr['t1'],option_attr['t2'])
    

    def logLike(self, coefs , w = 1):

        all_args = self.fixed_args | dict(zip(self.param_keys, coefs))

        p = self.choiceProb(all_args)
        y = self.choice
        
        logLogit = y*np.log(p) + (1-y)*np.log(1-p)

        if len(logLogit.shape) > 1:
            logLogit = logLogit.sum(axis = 1)
        
        if isinstance(w,int) and w == 1:
            w = [w]*len(y)

        return logLogit @ w
        

    def fit_param(self,bounds=None):

        obj = lambda coefs: - self.logLike(coefs)*1000

        x0 = list(self.param_init)

        self.result = optimize.minimize(obj, x0, method='L-BFGS-B', bounds=bounds)

        return self.result
    
    

class mixedModel(choiceModel):

    def set_init_mixed(self, latent_class: dict, latent_share: list):
        self.latent_class = latent_class
        self.latent_share = latent_share
        self.param_keys = list(latent_class.keys())
        self.n_class = len(list(latent_class.values())[0])
        self.obj_values = [0]*self.n_class



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

            obj = lambda coefs: - self.logLike(coefs,w) * 1000

            result = optimize.minimize(obj, param_init, method='L-BFGS-B', bounds=bounds)

            for key,value in zip(self.latent_class.keys(),result.x):
                self.latent_class[key][c] = value
            
            self.obj_values[c] = result.fun

            if np.isnan(result.fun):
                self.obj_values[c] = obj(result.x)

        return self.obj_values
    
    
    def runEM(self,max_iter=500,tol=1e-4,bounds=None):

        # expectation-maximization (EM) algorithm
        self.updateShare()
        old_obj = self.obj_values
        new_obj = self.fitEachClass(bounds)

        new_share = np.round(self.latent_share, 3)
        new_param = dict(zip(self.param_keys, 
                                np.round(list(self.latent_class.values()), 3)))
        i = 1

        print('Iteration', i)
        print('Class share', new_share)
        print('Class parameter', new_param)
        print('objective', np.round(new_obj,3))

        while i <= max_iter and all(
            abs(o - h) < tol for o,h in zip(old_obj, new_obj)
        ):
            self.updateShare()
            old_obj = self.obj_values
            new_obj = self.fitEachClass(bounds)

            new_share = np.round(self.latent_share, 3)
            new_param = dict(zip(self.param_keys, 
                                 np.round(list(self.latent_class.values()), 3)))
            
            i += 1

            print('Iteration', i)
            print('Class share', new_share)
            print('Class parameter', new_param)
            print('objective', np.round(new_obj,3))

            


        



    
    


        














    